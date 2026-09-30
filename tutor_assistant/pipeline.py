"""Прогон пайплайна на датасете: извлечение, классификация, зоны, флаги.

Флаги постконтроля независимы и считаются отдельно:

* «требуется уточнение» — запись не понята: unknown в classify или в
  check_zone нет зоны и нет уверенного вывода;
* «низкая уверенность» — ответ есть, но уверенность ниже
  config.LOW_CONFIDENCE_THRESHOLD.
"""

import json
from pathlib import Path
from typing import Any

from tutor_assistant import config
from tutor_assistant.classify import ClassificationResult, classify_detailed
from tutor_assistant.extract import FIELDS, ExtractedValue, extract
from tutor_assistant.zones import ZoneResult, check_zone_detailed

STATUS_CORRECT = "correct"
STATUS_CORRECT_NONE = "correct_none"
STATUS_MISSED = "missed"
STATUS_WRONG = "wrong"
STATUSES: tuple[str, ...] = (
    STATUS_CORRECT,
    STATUS_CORRECT_NONE,
    STATUS_MISSED,
    STATUS_WRONG,
)
NEEDS_CLARIFICATION = "needs_clarification"
LOW_CONFIDENCE = "low_confidence"
FLAGS: tuple[str, ...] = (NEEDS_CLARIFICATION, LOW_CONFIDENCE)


def load_json(path: Path) -> Any:
    """Прочитать JSON-файл.

    Args:
        path: Путь к файлу.

    Returns:
        Разобранное содержимое.
    """
    with path.open(encoding=config.FILE_ENCODING) as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    """Записать данные в JSON-файл без экранирования кириллицы.

    Args:
        data: Сериализуемые данные.
        path: Путь к файлу; каталог создаётся при необходимости.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=config.FILE_ENCODING, newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=config.JSON_INDENT)
        file.write("\n")


def field_status(actual: ExtractedValue, expected: ExtractedValue) -> str:
    """Сравнить извлечённое значение поля с разметкой.

    Args:
        actual: Значение из extract.
        expected: Ожидаемое значение из expected.json.

    Returns:
        correct, correct_none (верно вернули None), missed или wrong.
    """
    if actual == expected:
        return STATUS_CORRECT_NONE if actual is None else STATUS_CORRECT
    if actual is None:
        return STATUS_MISSED
    return STATUS_WRONG


def classification_flags(result: ClassificationResult) -> dict[str, bool]:
    """Посчитать флаги постконтроля для классификации записи.

    Args:
        result: Подробный результат classify.

    Returns:
        Словарь флагов needs_clarification и low_confidence.
    """
    unknown = result.label == config.UNKNOWN_TYPE
    uncertain = result.confidence < config.LOW_CONFIDENCE_THRESHOLD
    return {
        NEEDS_CLARIFICATION: unknown,
        LOW_CONFIDENCE: not unknown and uncertain,
    }


def zone_flags(result: ZoneResult) -> dict[str, bool]:
    """Посчитать флаги постконтроля для проверки зоны.

    Args:
        result: Подробный результат check_zone.

    Returns:
        Словарь флагов needs_clarification и low_confidence.
    """
    uncertain = result.confidence < config.LOW_CONFIDENCE_THRESHOLD
    return {
        NEEDS_CLARIFICATION: not result.is_zone and uncertain,
        LOW_CONFIDENCE: result.is_zone and uncertain,
    }


def process_record(
    record: dict[str, str], gold: dict[str, Any]
) -> dict[str, Any]:
    """Обработать запись R1–R6: extract, classify, сверка и флаги.

    Args:
        record: Запись с полями id и text.
        gold: Ожидаемые значения для записи.

    Returns:
        Результат обработки записи для results.json.
    """
    fields = extract(record["text"])
    result = classify_detailed(record["text"])
    return {
        "id": record["id"],
        "text": record["text"],
        "extract": fields,
        "expected": {field: gold[field] for field in FIELDS},
        "field_status": {
            field: field_status(fields[field], gold[field])
            for field in FIELDS
        },
        "classify": {
            "type": result.label,
            "confidence": result.confidence,
            "scores": result.scores,
            "shares": result.shares,
            "leader": result.leader,
            "runner_up": result.runner_up,
            "gap": result.gap,
            "evidence": result.evidence,
            "expected_type": gold["type"],
        },
        "flags": classification_flags(result),
    }


def process_observation(
    index: int,
    note: str,
    gold: dict[str, Any],
    use_llm: bool,
    client: Any | None,
) -> dict[str, Any]:
    """Обработать наблюдение: check_zone, сверка и флаги.

    Args:
        index: Номер наблюдения, начиная с 1.
        note: Текст наблюдения.
        gold: Ожидаемый ответ из expected.json.
        use_llm: Пробовать ли LLM.
        client: Клиент Anthropic или заглушка.

    Returns:
        Результат обработки наблюдения для results.json.
    """
    result = check_zone_detailed(note, use_llm=use_llm, client=client)
    zone_ok = not result.is_zone or result.zone in gold["zones"]
    flags = zone_flags(result)
    return {
        "index": index,
        "note": note,
        "is_zone": result.is_zone,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "zone": result.zone,
        "source": result.source,
        "scores": result.scores,
        "doubts": list(result.doubts),
        "expected": gold,
        "matches_expected": result.is_zone == gold["is_zone"] and zone_ok,
        "flagged": any(flags.values()),
        "flags": flags,
    }


def extraction_summary(
    records: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    """Посчитать по каждому полю, сколько значений верно, пропущено, ошибочно.

    Args:
        records: Обработанные записи.

    Returns:
        Счётчики статусов по полям.
    """
    summary = {field: dict.fromkeys(STATUSES, 0) for field in FIELDS}
    for record in records:
        for field, status in record["field_status"].items():
            summary[field][status] += 1
    return summary


def min_keyword_weight() -> float:
    """Вернуть минимальный вес признака классификации из конфига.

    Returns:
        Наименьший вес в config.CLASSIFY_KEYWORDS.
    """
    return min(
        weight
        for rules in config.CLASSIFY_KEYWORDS.values()
        for weight in rules.values()
    )


def extra_words_to_unknown(
    scores: dict[str, float],
    leader: str,
    runner_up: str,
    threshold: float,
    weight: float,
) -> int:
    """Посчитать, сколько лишних признаков сбивают ответ классификатора.

    К скору второго места по одному добавляются признаки веса weight,
    пока разрыв долей не станет меньше порога или лидер не сменится.

    Args:
        scores: Сырые скоры типов записи.
        leader: Тип на первом месте.
        runner_up: Тип на втором месте.
        threshold: Порог разрыва для unknown.
        weight: Вес одного лишнего признака.

    Returns:
        Число лишних признаков, после которого ответ перестаёт быть
        прежним (0 — запись уже unknown).
    """
    boosted = dict(scores)
    words = 0
    while boosted[runner_up] < boosted[leader]:
        total = sum(boosted.values())
        leader_share = round(boosted[leader] / total, config.SCORE_DIGITS)
        runner_share = round(boosted[runner_up] / total, config.SCORE_DIGITS)
        gap = round(leader_share - runner_share, config.SCORE_DIGITS)
        if gap < threshold:
            break
        boosted[runner_up] += weight
        words += 1
    return words


def threshold_sweep(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Перебрать пороги unknown и посчитать их последствия на записях.

    Правило подбора: подходит порог, при котором ни одна верно
    классифицированная запись не становится unknown, даже если добавить
    в неё config.UNKNOWN_NOISE_TOLERANCE_WORDS лишних признаков
    минимального веса в пользу второго места. Из подходящих выбирается
    наибольший: спорные записи лучше отдавать человеку.

    Args:
        records: Обработанные записи.

    Returns:
        Для каждого порога: какие записи стали бы unknown, сколько
        записей классифицировано верно, минимальный запас в лишних
        признаках и признак выбора по правилу.
    """
    weight = min_keyword_weight()
    correct_records = [
        record for record in records
        if record["classify"]["leader"] == record["classify"]["expected_type"]
    ]
    rows: list[dict[str, Any]] = []
    for threshold in config.THRESHOLD_SWEEP:
        unknown = [
            record["id"] for record in records
            if record["classify"]["gap"] < threshold
        ]
        correct = sum(
            record["classify"]["gap"] >= threshold
            for record in correct_records
        )
        margins = [
            extra_words_to_unknown(
                record["classify"]["scores"],
                record["classify"]["leader"],
                record["classify"]["runner_up"],
                threshold,
                weight,
            )
            for record in correct_records
        ]
        min_margin = min(margins, default=0)
        rows.append(
            {
                "threshold": threshold,
                "unknown": unknown,
                "correct": correct,
                "min_extra_words": min_margin,
                "passes_rule": (
                    correct == len(correct_records)
                    and min_margin > config.UNKNOWN_NOISE_TOLERANCE_WORDS
                ),
            }
        )
    passing = [row["threshold"] for row in rows if row["passes_rule"]]
    selected = max(passing, default=None)
    for row in rows:
        row["selected_by_rule"] = row["threshold"] == selected
    return rows


def _share(count: int, total: int) -> float:
    """Посчитать долю с округлением.

    Args:
        count: Числитель.
        total: Знаменатель.

    Returns:
        Доля или 0.0 при пустом знаменателе.
    """
    return round(count / total, config.SCORE_DIGITS) if total else 0.0


def flag_summary(
    records: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Посчитать доли флагов отдельно для записей, наблюдений и всего.

    Args:
        records: Обработанные записи R1–R6.
        observations: Обработанные наблюдения.

    Returns:
        Для каждого флага: состав, количества и доли.
    """
    total = len(records) + len(observations)
    summary: dict[str, dict[str, Any]] = {}
    for flag in FLAGS:
        record_ids = [item["id"] for item in records if item["flags"][flag]]
        note_ids = [
            f"#{item['index']}"
            for item in observations
            if item["flags"][flag]
        ]
        count = len(record_ids) + len(note_ids)
        summary[flag] = {
            "records": record_ids,
            "observations": note_ids,
            "records_share": _share(len(record_ids), len(records)),
            "observations_share": _share(len(note_ids), len(observations)),
            "count": count,
            "total": total,
            "share": _share(count, total),
        }
    return summary


def zones_summary(observations: list[dict[str, Any]]) -> dict[str, int]:
    """Сверить ответы check_zone с разметкой.

    Args:
        observations: Обработанные наблюдения.

    Returns:
        Счётчики совпадений bool и зоны, а также флагов у спорных случаев.
    """
    disputed = [item for item in observations if item["expected"]["disputed"]]
    return {
        "total": len(observations),
        "bool_matches": sum(
            item["is_zone"] == item["expected"]["is_zone"]
            for item in observations
        ),
        "full_matches": sum(item["matches_expected"] for item in observations),
        "disputed": len(disputed),
        "disputed_flagged": sum(item["flagged"] for item in disputed),
        "clear_flagged": sum(
            item["flagged"] for item in observations
            if not item["expected"]["disputed"]
        ),
    }


def run_pipeline(
    dataset: dict[str, Any],
    expected: dict[str, Any],
    use_llm: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Прогнать все части пайплайна на датасете.

    Args:
        dataset: Содержимое dataset.json.
        expected: Содержимое expected.json.
        use_llm: Пробовать ли LLM в check_zone.
        client: Клиент Anthropic или заглушка.

    Returns:
        Полный результат для results.json и отчёта.

    Raises:
        ValueError: Если разметка не соответствует датасету.
    """
    gold_records = {item["id"]: item for item in expected["records"]}
    gold_notes = [item["note"] for item in expected["observations"]]
    if set(gold_records) != {item["id"] for item in dataset["records"]}:
        raise ValueError("expected.json не соответствует записям dataset.json")
    if gold_notes != dataset["observations"]:
        raise ValueError(
            "expected.json не соответствует наблюдениям dataset.json"
        )
    records = [
        process_record(record, gold_records[record["id"]])
        for record in dataset["records"]
    ]
    observations = [
        process_observation(index, note, gold, use_llm, client)
        for index, (note, gold) in enumerate(
            zip(dataset["observations"], expected["observations"]), start=1
        )
    ]
    classified_ok = sum(
        record["classify"]["type"] == record["classify"]["expected_type"]
        for record in records
    )
    return {
        "settings": {
            "low_confidence_threshold": config.LOW_CONFIDENCE_THRESHOLD,
            "unknown_gap_threshold": config.UNKNOWN_GAP_THRESHOLD,
            "llm_requested": use_llm,
            "llm_model": config.LLM_MODEL,
            "zone_sources": sorted({item["source"] for item in observations}),
        },
        "records": records,
        "observations": observations,
        "summary": {
            "extraction": extraction_summary(records),
            "classification_correct": classified_ok,
            "threshold_sweep": threshold_sweep(records),
            "zones": zones_summary(observations),
            "flags": flag_summary(records, observations),
        },
    }
