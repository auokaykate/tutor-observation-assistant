"""Тесты пайплайна: датасет, флаги постконтроля и отчёты."""

from typing import Any

import pytest

from tutor_assistant import config
from tutor_assistant.pipeline import (
    LOW_CONFIDENCE,
    NEEDS_CLARIFICATION,
    load_json,
    run_pipeline,
    zone_flags,
)
from tutor_assistant.report import build_results_markdown, render_report
from tutor_assistant.zones import ZoneResult

RECORD_IDS = ["R1", "R2", "R3", "R4", "R5", "R6"]
OBSERVATIONS_IN_PDF = 15


@pytest.fixture(scope="module")
def results(
    dataset: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Прогнать пайплайн на датасете в режиме правил."""
    return run_pipeline(dataset, expected)


def _template() -> str:
    """Прочитать шаблон RESULTS.md.

    Returns:
        Текст шаблона.
    """
    return config.RESULTS_TEMPLATE_PATH.read_text(
        encoding=config.FILE_ENCODING
    )


def test_dataset_is_verbatim_copy(dataset: dict[str, Any]) -> None:
    """В датасете R1–R6 и все наблюдения, особенности текста сохранены."""
    texts = {record["id"]: record["text"] for record in dataset["records"]}
    assert list(texts) == RECORD_IDS
    assert len(dataset["observations"]) == OBSERVATIONS_IN_PDF
    assert "Ребёнок CH0107, зона: моторика." in texts["R2"]
    assert texts["R3"].startswith("03/01/25, мама ребёнка CH-0342:")
    assert "сон~390 минут,настрй норм" in texts["R4"]


def test_expected_matches_dataset(
    dataset: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Разметка относится ровно к тем же записям и наблюдениям."""
    assert [item["id"] for item in expected["records"]] == RECORD_IDS
    notes = [item["note"] for item in expected["observations"]]
    assert notes == dataset["observations"]


def test_flags_are_counted_separately(results: dict[str, Any]) -> None:
    """Флаги считаются отдельно и не ставятся одному элементу вместе."""
    flags = results["summary"]["flags"]
    assert flags[NEEDS_CLARIFICATION]["records"] == []
    assert flags[NEEDS_CLARIFICATION]["observations"] == [
        "#9", "#10", "#11", "#12",
    ]
    assert flags[LOW_CONFIDENCE]["records"] == ["R6"]
    assert flags[LOW_CONFIDENCE]["observations"] == [
        "#2", "#13", "#14", "#15",
    ]
    items = results["records"] + results["observations"]
    assert not any(all(item["flags"].values()) for item in items)


def test_flag_shares(results: dict[str, Any]) -> None:
    """Доли флагов считаются от всех записей и наблюдений."""
    total = len(results["records"]) + len(results["observations"])
    for data in results["summary"]["flags"].values():
        assert data["total"] == total
        assert data["count"] == (
            len(data["records"]) + len(data["observations"])
        )
        assert data["share"] == round(data["count"] / total,
                                      config.SCORE_DIGITS)


@pytest.mark.parametrize(
    ("is_zone", "confidence", "needs_clarification", "low_confidence"),
    [
        (False, config.NON_ZONE_CONFIDENCE, True, False),
        (False, config.LOW_CONFIDENCE_THRESHOLD, False, False),
        (True, config.NON_ZONE_CONFIDENCE, False, True),
        (True, config.ZONE_MAX_CONFIDENCE, False, False),
    ],
    ids=["no_zone_uncertain", "no_zone_confident", "zone_uncertain",
         "zone_confident"],
)
def test_zone_flag_rules(
    is_zone: bool,
    confidence: float,
    needs_clarification: bool,
    low_confidence: bool,
) -> None:
    """«Требуется уточнение» — нет зоны и нет уверенности; «низкая
    уверенность» — зона есть, но уверенность ниже порога."""
    result = ZoneResult(
        is_zone, confidence, "объяснение", None, config.SOURCE_RULES, {}, ()
    )
    assert zone_flags(result) == {
        NEEDS_CLARIFICATION: needs_clarification,
        LOW_CONFIDENCE: low_confidence,
    }


def test_all_parts_match_annotation(results: dict[str, Any]) -> None:
    """Извлечение, классификация и зоны совпадают с разметкой."""
    summary = results["summary"]
    assert all(
        counts["missed"] == 0 and counts["wrong"] == 0
        for counts in summary["extraction"].values()
    )
    assert summary["classification_correct"] == len(results["records"])
    zones = summary["zones"]
    assert zones["full_matches"] == zones["total"]
    assert zones["disputed_flagged"] == zones["disputed"]
    assert zones["clear_flagged"] == 0


def test_results_markdown_is_built_from_report(
    results: dict[str, Any],
) -> None:
    """RESULTS.md собирается из отчёта без незаполненных подстановок."""
    report = render_report(results)
    results_md = build_results_markdown(report, _template())
    assert "$" not in results_md
    for section in (
        "## Часть 1. Извлечение полей (R1–R6)",
        "## Часть 2. Классификация типа записи (R1–R6)",
        "## Часть 3. Проверка зон развития (наблюдения)",
        "## Флаги постконтроля",
        "## Подходы и компромиссы",
    ):
        assert section in results_md


def test_committed_outputs_are_up_to_date(results: dict[str, Any]) -> None:
    """output/report.md и RESULTS.md соответствуют текущему коду.

    Если отчёт получен в LLM-режиме, сравнение с правилами пропускается.
    """
    report_path = config.OUTPUT_DIR / config.REPORT_FILENAME
    results_path = config.OUTPUT_DIR / config.RESULTS_JSON_FILENAME
    if not report_path.exists() or not results_path.exists():
        pytest.skip("отчёт ещё не сформирован: запустите python main.py")
    if load_json(results_path)["settings"]["llm_requested"]:
        pytest.skip("отчёт сформирован в LLM-режиме")
    report = report_path.read_text(encoding=config.FILE_ENCODING)
    assert report == render_report(results)
    results_md = config.RESULTS_MD_PATH.read_text(
        encoding=config.FILE_ENCODING
    )
    assert results_md == build_results_markdown(report, _template())
