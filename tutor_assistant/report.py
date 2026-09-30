"""Отчёт по прогону: output/report.md и сборка RESULTS.md из шаблона.

Все числа в отчёте берутся из результатов прогона и из конфига. RESULTS.md
собирается из output/report.md и шаблона results_template.md, в котором
чисел нет: значения порогов подставляются из config.
"""

from pathlib import Path
from string import Template
from typing import Any

from tutor_assistant import config
from tutor_assistant.extract import FIELDS, extract_date
from tutor_assistant.pipeline import (
    FLAGS,
    LOW_CONFIDENCE,
    NEEDS_CLARIFICATION,
    STATUS_CORRECT,
    STATUS_CORRECT_NONE,
    STATUS_MISSED,
    STATUS_WRONG,
    min_keyword_weight,
)

STATUS_TITLES: dict[str, str] = {
    STATUS_CORRECT: "верно",
    STATUS_CORRECT_NONE: "верно (None)",
    STATUS_MISSED: "пропущено",
    STATUS_WRONG: "ошибка",
}
FLAG_TITLES: dict[str, str] = {
    NEEDS_CLARIFICATION: config.FLAG_NEEDS_CLARIFICATION,
    LOW_CONFIDENCE: config.FLAG_LOW_CONFIDENCE,
}
NO_FLAGS = "—"
NOTHING = "нет"
SLASH_DATE_EXAMPLE = "03/01/25"
NEWLINE = "\n"


def _cell(value: Any) -> str:
    """Отформатировать значение для ячейки markdown-таблицы.

    Args:
        value: Любое значение.

    Returns:
        Строка; None выводится как «None», символ «|» экранируется.
    """
    if value is None:
        return "None"
    return str(value).replace("|", "\\|")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Собрать markdown-таблицу.

    Args:
        headers: Заголовки столбцов.
        rows: Строки таблицы.

    Returns:
        Строки таблицы.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in rows
    )
    return lines


def _percent(share: float) -> str:
    """Перевести долю в проценты.

    Args:
        share: Доля от 0 до 1.

    Returns:
        Строка вида «19.0%».
    """
    value = share * config.PERCENT_MULTIPLIER
    return f"{value:.{config.PERCENT_DIGITS}f}%"


def _flags_text(flags: dict[str, bool]) -> str:
    """Перечислить активные флаги.

    Args:
        flags: Флаги постконтроля.

    Returns:
        Названия активных флагов или прочерк.
    """
    active = [FLAG_TITLES[flag] for flag in FLAGS if flags[flag]]
    return ", ".join(active) if active else NO_FLAGS


def _join(items: list[str]) -> str:
    """Перечислить элементы через запятую.

    Args:
        items: Элементы списка.

    Returns:
        Строка с элементами или «нет».
    """
    return ", ".join(items) if items else NOTHING


def _status_cell(record: dict[str, Any], field: str) -> str:
    """Описать результат сверки поля с разметкой.

    Args:
        record: Обработанная запись.
        field: Имя поля.

    Returns:
        Текст статуса с ожидаемым значением для пропусков и ошибок.
    """
    status = record["field_status"][field]
    title = STATUS_TITLES[status]
    expected = _cell(record["expected"][field])
    if status == STATUS_MISSED:
        return f"{title} (ожидалось {expected})"
    if status == STATUS_WRONG:
        return f"{title}: {_cell(record['extract'][field])} вместо {expected}"
    return title


def render_extraction(results: dict[str, Any]) -> list[str]:
    """Отрисовать раздел Части 1.

    Args:
        results: Результат run_pipeline.

    Returns:
        Строки раздела.
    """
    records = results["records"]
    summary = results["summary"]["extraction"]
    lines = [
        "## Часть 1. Извлечение полей (R1–R6)",
        "",
        "### Что извлечено",
        "",
    ]
    lines += _table(
        ["ID", *FIELDS],
        [
            [record["id"], *(record["extract"][field] for field in FIELDS)]
            for record in records
        ],
    )
    lines += ["", "### Сверка с ручной разметкой (expected.json)", ""]
    lines += _table(
        ["ID", *FIELDS],
        [
            [record["id"], *(_status_cell(record, field) for field in FIELDS)]
            for record in records
        ],
    )
    lines += ["", "### Итог по полям", ""]
    rows: list[list[Any]] = []
    totals = dict.fromkeys(STATUS_TITLES, 0)
    for field in FIELDS:
        counts = summary[field]
        correct = counts[STATUS_CORRECT] + counts[STATUS_CORRECT_NONE]
        rows.append(
            [
                field,
                counts[STATUS_CORRECT],
                counts[STATUS_CORRECT_NONE],
                counts[STATUS_MISSED],
                counts[STATUS_WRONG],
                f"{correct} из {len(records)}",
            ]
        )
        for status in totals:
            totals[status] += counts[status]
    all_correct = totals[STATUS_CORRECT] + totals[STATUS_CORRECT_NONE]
    rows.append(
        [
            "итого",
            totals[STATUS_CORRECT],
            totals[STATUS_CORRECT_NONE],
            totals[STATUS_MISSED],
            totals[STATUS_WRONG],
            f"{all_correct} из {len(records) * len(FIELDS)}",
        ]
    )
    lines += _table(
        [
            "Поле",
            "Корректно (значение)",
            "Корректно (None)",
            "Пропущено",
            "Ошибочно",
            "Всего корректно",
        ],
        rows,
    )
    return lines


def _place(classification: dict[str, Any], key: str) -> str:
    """Описать тип на заданном месте с его долей.

    Args:
        classification: Раздел classify обработанной записи.
        key: leader или runner_up.

    Returns:
        Строка вида «observation (0.8)» или «None».
    """
    record_type = classification[key]
    if record_type is None:
        return "None"
    return f"{record_type} ({classification['shares'][record_type]})"


def render_threshold(results: dict[str, Any]) -> list[str]:
    """Отрисовать обоснование порога unknown на данных прогона.

    Args:
        results: Результат run_pipeline.

    Returns:
        Строки подраздела.
    """
    records = results["records"]
    sweep = results["summary"]["threshold_sweep"]
    threshold = config.UNKNOWN_GAP_THRESHOLD
    weight = min_keyword_weight()
    tolerance = config.UNKNOWN_NOISE_TOLERANCE_WORDS
    correct_records = [
        record for record in records
        if record["classify"]["leader"] == record["classify"]["expected_type"]
    ]
    gaps = ", ".join(
        f"{record['id']} — {record['classify']['gap']}" for record in records
    )
    lines = [
        "### Подбор порога unknown",
        "",
        f"Порог в конфиге: `UNKNOWN_GAP_THRESHOLD = {threshold}`. "
        f"Разрывы долей первого и второго места: {gaps}.",
        "",
    ]
    lines += _table(
        [
            "Порог",
            "Станут unknown",
            f"Верных ответов сохранится (из {len(correct_records)})",
            f"Мин. запас: лишних признаков веса {weight}",
            "Подходит по правилу",
            "Выбран",
        ],
        [
            [
                row["threshold"],
                _join(row["unknown"]),
                row["correct"],
                row["min_extra_words"],
                "да" if row["passes_rule"] else "нет",
                "да" if row["selected_by_rule"] else "",
            ]
            for row in sweep
        ],
    )
    selected = next(
        (row["threshold"] for row in sweep if row["selected_by_rule"]), None
    )
    lines += [
        "",
        "Правило подбора: берём наибольший порог из сетки, при котором ни "
        "одна верно классифицированная запись не уходит в unknown и "
        "выдерживает случайные слова другого типа. Запас в таблице — "
        f"сколько признаков веса {weight} в пользу второго места нужно "
        "добавить в запись, чтобы ответ сбился; по правилу он должен быть "
        f"больше {tolerance} (`UNKNOWN_NOISE_TOLERANCE_WORDS`). Из "
        "подходящих порогов берётся наибольший: спорную запись лучше "
        "отдать человеку, чем угадать.",
        "",
    ]
    if correct_records:
        closest = min(
            correct_records, key=lambda record: record["classify"]["gap"]
        )
        info = closest["classify"]
        lines.append(
            f"- Ближе всех к границе {closest['id']}: "
            f"{_place(info, 'leader')} против {_place(info, 'runner_up')}, "
            f"разрыв {info['gap']}."
        )
    first_fail = next(
        (row for row in sweep if not row["passes_rule"]), None
    )
    if first_fail is not None:
        lines.append(
            f"- С порога {first_fail['threshold']} правило нарушается: "
            f"верных ответов сохраняется {first_fail['correct']} из "
            f"{len(correct_records)}, минимальный запас — "
            f"{first_fail['min_extra_words']}."
        )
    verdict = "совпадает" if selected == threshold else "не совпадает"
    lines.append(
        f"- По правилу выбран порог {selected}; в конфиге {threshold} — "
        f"{verdict}."
    )
    lines.append(
        f"- При пороге {threshold} доля лидера должна превышать долю "
        f"второго места не меньше чем на {_percent(threshold)} суммарного "
        "скора; при меньшем перевесе запись считается ничьей и уходит на "
        "уточнение."
    )
    unknown_now = [
        record["id"] for record in records
        if record["classify"]["type"] == config.UNKNOWN_TYPE
    ]
    lines.append(
        f"- Записей unknown при текущем пороге: {_join(unknown_now)}. "
        "Срабатывание порога проверено тестами на текстах со смешанными "
        "признаками. Записей мало, поэтому на реальном потоке порог нужно "
        "перепроверить."
    )
    return lines


def render_classification(results: dict[str, Any]) -> list[str]:
    """Отрисовать раздел Части 2.

    Args:
        results: Результат run_pipeline.

    Returns:
        Строки раздела.
    """
    records = results["records"]
    rows: list[list[Any]] = []
    for record in records:
        info = record["classify"]
        rows.append(
            [
                record["id"],
                *(info["scores"][kind] for kind in config.RECORD_TYPES),
                _place(info, "leader"),
                _place(info, "runner_up"),
                info["gap"],
                info["type"],
                info["confidence"],
                info["expected_type"],
                _flags_text(record["flags"]),
            ]
        )
    lines = ["## Часть 2. Классификация типа записи (R1–R6)", ""]
    lines += _table(
        [
            "ID",
            *config.RECORD_TYPES,
            "1-е место (доля)",
            "2-е место (доля)",
            "Разрыв",
            "Тип",
            "Уверенность",
            "Разметка",
            "Флаги",
        ],
        rows,
    )
    correct = results["summary"]["classification_correct"]
    lines += [
        "",
        f"Совпадение с разметкой: {correct} из {len(records)}.",
        "",
        "Сработавшие признаки (фрагменты исходного текста):",
        "",
    ]
    for record in records:
        parts = [
            f"{kind}: " + ", ".join(f"«{item}»" for item in items)
            for kind, items in record["classify"]["evidence"].items()
            if items
        ]
        lines.append(f"- {record['id']}: " + ("; ".join(parts) or NOTHING))
    lines += ["", *render_threshold(results)]
    return lines


def render_zones(results: dict[str, Any]) -> list[str]:
    """Отрисовать раздел Части 3.

    Args:
        results: Результат run_pipeline.

    Returns:
        Строки раздела.
    """
    observations = results["observations"]
    summary = results["summary"]["zones"]
    sources = ", ".join(results["settings"]["zone_sources"])
    lines = [
        "## Часть 3. Проверка зон развития (наблюдения)",
        "",
        f"Источник ответов: {sources}. Порог низкой уверенности: "
        f"{config.LOW_CONFIDENCE_THRESHOLD}.",
        "",
    ]
    lines += _table(
        [
            "#",
            "Наблюдение",
            "Ответ",
            "Зона",
            "Уверенность",
            "Источник",
            "Флаги",
            "Объяснение",
        ],
        [
            [
                item["index"],
                item["note"],
                item["is_zone"],
                item["zone"],
                item["confidence"],
                item["source"],
                _flags_text(item["flags"]),
                item["explanation"],
            ]
            for item in observations
        ],
    )
    clear_total = summary["total"] - summary["disputed"]
    lines += [
        "",
        f"Сверка с разметкой: True/False совпал в {summary['bool_matches']} "
        f"из {summary['total']}, ответ вместе с зоной — в "
        f"{summary['full_matches']} из {summary['total']} (для спорных "
        "случаев подходит любая из размеченных зон). Флаг постконтроля "
        f"получили {summary['disputed_flagged']} из {summary['disputed']} "
        f"спорных случаев и {summary['clear_flagged']} из {clear_total} "
        "однозначных.",
        "",
        "### Спорные случаи",
        "",
        "Спорный случай не угадывается. Если есть признаки конкурирующей "
        "зоны или названо ситуативное обстоятельство (погода, праздник), "
        f"уверенность умножается на {config.ZONE_DOUBT_FACTOR} за каждый вид "
        "сомнения, и результат опускается ниже порога "
        f"{config.LOW_CONFIDENCE_THRESHOLD}. Если признаков зон нет, а "
        "описано только организационное событие, возвращается False с "
        f"уверенностью {config.NON_ZONE_CONFIDENCE}: реакция ребёнка не "
        "описана, и запись уходит на уточнение.",
        "",
    ]
    lines += _table(
        ["#", "Наблюдение", "Ответ", "Зона", "Уверенность", "Флаги",
         "Сомнения"],
        [
            [
                item["index"],
                item["note"],
                item["is_zone"],
                item["zone"],
                item["confidence"],
                _flags_text(item["flags"]),
                "; ".join(item["doubts"]) or NOTHING,
            ]
            for item in observations
            if item["expected"]["disputed"]
        ],
    )
    return lines


def _describe_record(record: dict[str, Any]) -> str:
    """Кратко описать запись с флагом.

    Args:
        record: Обработанная запись.

    Returns:
        Строка с типом, уверенностью, разрывом и суммой весов признаков.
    """
    info = record["classify"]
    total = sum(info["scores"].values())
    return (
        f"{record['id']} — тип {info['type']}, уверенность "
        f"{info['confidence']}, разрыв {info['gap']}, сумма весов "
        f"признаков {total} при насыщении "
        f"{config.CLASSIFY_EVIDENCE_SATURATION}"
    )


def _describe_observation(item: dict[str, Any]) -> str:
    """Кратко описать наблюдение с флагом.

    Args:
        item: Обработанное наблюдение.

    Returns:
        Строка с зоной, уверенностью и сомнениями.
    """
    zone = item["zone"] or "зона не найдена"
    doubts = "; ".join(item["doubts"]) or NOTHING
    return (
        f"#{item['index']} «{item['note']}» — {zone}, уверенность "
        f"{item['confidence']} ({doubts})"
    )


def render_flags(results: dict[str, Any]) -> list[str]:
    """Отрисовать раздел о флагах постконтроля.

    Args:
        results: Результат run_pipeline.

    Returns:
        Строки раздела.
    """
    records = results["records"]
    observations = results["observations"]
    flags = results["summary"]["flags"]
    threshold = config.LOW_CONFIDENCE_THRESHOLD
    lines = [
        "## Флаги постконтроля",
        "",
        f"- **{config.FLAG_NEEDS_CLARIFICATION}** — запись не понята: "
        "unknown в classify или в check_zone нет зоны и нет уверенного "
        f"вывода (False с уверенностью ниже {threshold}).",
        f"- **{config.FLAG_LOW_CONFIDENCE}** — ответ есть (тип записи или "
        f"зона), но уверенность ниже {threshold} "
        "(`LOW_CONFIDENCE_THRESHOLD`).",
        "",
        "Флаги считаются независимо. Для одного элемента они не "
        "пересекаются по построению: первый ставится, когда ответа нет, "
        "второй — когда ответ есть.",
        "",
    ]
    rows: list[list[Any]] = []
    for flag in FLAGS:
        data = flags[flag]
        rows.append(
            [
                FLAG_TITLES[flag],
                f"{len(data['records'])} из {len(records)} "
                f"({_percent(data['records_share'])})",
                f"{len(data['observations'])} из {len(observations)} "
                f"({_percent(data['observations_share'])})",
                f"{data['count']} из {data['total']}",
                _percent(data["share"]),
            ]
        )
    lines += _table(
        [
            "Флаг",
            "Записи R1–R6 (classify)",
            "Наблюдения (check_zone)",
            "Всего",
            "Доля",
        ],
        rows,
    )
    lines += ["", "### Состав", ""]
    for flag in FLAGS:
        flagged = [
            _describe_record(record)
            for record in records if record["flags"][flag]
        ]
        flagged += [
            _describe_observation(item)
            for item in observations if item["flags"][flag]
        ]
        lines.append(f"**{FLAG_TITLES[flag]}** ({len(flagged)}):")
        lines.append("")
        lines += [f"- {entry}" for entry in flagged] or [f"- {NOTHING}"]
        lines.append("")
    return lines


def render_report(results: dict[str, Any]) -> str:
    """Собрать полный текст output/report.md.

    Args:
        results: Результат run_pipeline.

    Returns:
        Markdown-текст отчёта.
    """
    command = "python main.py"
    if results["settings"]["llm_requested"]:
        command += " --llm"
    lines = [
        "# Отчёт о прогоне пайплайна",
        "",
        f"Сформирован командой `{command}` по dataset.json. Сверка — с "
        "ручной разметкой expected.json.",
        "",
        *render_extraction(results),
        "",
        *render_classification(results),
        "",
        *render_zones(results),
        "",
        *render_flags(results),
    ]
    return NEWLINE.join(lines).rstrip() + NEWLINE


def template_values() -> dict[str, str]:
    """Подготовить значения из конфига для шаблона RESULTS.md.

    Returns:
        Словарь подстановок для string.Template.
    """
    doubt_cap = round(
        config.ZONE_MAX_CONFIDENCE * config.ZONE_DOUBT_FACTOR,
        config.CONFIDENCE_DIGITS,
    )
    values: dict[str, Any] = {
        "low_threshold": config.LOW_CONFIDENCE_THRESHOLD,
        "unknown_threshold": config.UNKNOWN_GAP_THRESHOLD,
        "unknown_confidence": config.UNKNOWN_CONFIDENCE,
        "evidence_saturation": config.CLASSIFY_EVIDENCE_SATURATION,
        "classify_max": config.CLASSIFY_MAX_CONFIDENCE,
        "slash_example": SLASH_DATE_EXAMPLE,
        "slash_iso": extract_date(SLASH_DATE_EXAMPLE),
        "zone_max": config.ZONE_MAX_CONFIDENCE,
        "half_saturation": config.ZONE_EVIDENCE_HALF_SATURATION,
        "doubt_factor": config.ZONE_DOUBT_FACTOR,
        "doubt_cap": doubt_cap,
        "competing_ratio": config.COMPETING_ZONE_RATIO,
        "non_zone_confidence": config.NON_ZONE_CONFIDENCE,
        "no_signal_confidence": config.NO_SIGNAL_CONFIDENCE,
        "sleep_gap": config.SLEEP_CONTEXT_MAX_GAP,
        "llm_model": config.LLM_MODEL,
        "llm_effort": config.LLM_EFFORT,
    }
    return {key: str(value) for key, value in values.items()}


def build_results_markdown(report_text: str, template_text: str) -> str:
    """Собрать RESULTS.md из текста отчёта и шаблона.

    Заголовок первого уровня отчёта отбрасывается, остальное вставляется
    на место $report.

    Args:
        report_text: Содержимое output/report.md.
        template_text: Содержимое results_template.md.

    Returns:
        Текст RESULTS.md.
    """
    body = report_text.splitlines()
    if body and body[0].startswith("# "):
        body = body[1:]
    values = template_values()
    values["report"] = NEWLINE.join(body).strip()
    return Template(template_text).substitute(values).rstrip() + NEWLINE


def write_results_markdown(
    report_path: Path,
    results_path: Path,
    template_path: Path = config.RESULTS_TEMPLATE_PATH,
) -> None:
    """Прочитать output/report.md и записать RESULTS.md.

    Args:
        report_path: Путь к output/report.md.
        results_path: Путь к RESULTS.md.
        template_path: Путь к шаблону RESULTS.md.
    """
    report_text = report_path.read_text(encoding=config.FILE_ENCODING)
    template_text = template_path.read_text(encoding=config.FILE_ENCODING)
    results_path.write_text(
        build_results_markdown(report_text, template_text),
        encoding=config.FILE_ENCODING,
        newline=NEWLINE,
    )
