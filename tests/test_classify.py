"""Тесты Части 2: классификация типа записи и порог unknown."""

import math
from typing import Any

import pytest

from tutor_assistant import config
from tutor_assistant.classify import classify, classify_detailed
from tutor_assistant.pipeline import process_record, threshold_sweep

MIXED_TEXT = "Рекомендация по занятию"


@pytest.mark.parametrize(
    ("record_id", "record_type"),
    [
        ("R1", "observation"),
        ("R2", "lesson_report"),
        ("R3", "parent_note"),
        ("R4", "observation"),
        ("R5", "recommendation"),
        ("R6", "observation"),
    ],
)
def test_record_types(
    records: dict[str, str], record_id: str, record_type: str
) -> None:
    """Типы R1–R6 совпадают с разметкой, уверенность в [0, 1]."""
    label, confidence = classify(records[record_id])
    assert label == record_type
    assert 0.0 <= confidence <= 1.0


def test_tie_returns_unknown() -> None:
    """Равные признаки двух типов дают unknown с нулевой уверенностью."""
    result = classify_detailed("Мама передала рекомендацию")
    assert result.gap == 0.0
    assert result.as_tuple() == (
        config.UNKNOWN_TYPE, config.UNKNOWN_CONFIDENCE
    )


def test_gap_threshold_boundary() -> None:
    """Разрыв меньше порога даёт unknown, равный порогу — тип."""
    gap = classify_detailed(MIXED_TEXT).gap
    at_gap = classify_detailed(MIXED_TEXT, threshold=gap)
    assert at_gap.label == "recommendation"
    above = classify_detailed(
        MIXED_TEXT, threshold=math.nextafter(gap, math.inf)
    )
    assert above.label == config.UNKNOWN_TYPE
    assert above.confidence == config.UNKNOWN_CONFIDENCE


def test_default_threshold_comes_from_config(monkeypatch: Any) -> None:
    """Порог unknown по умолчанию берётся из config."""
    gap = classify_detailed(MIXED_TEXT).gap
    assert classify(MIXED_TEXT)[0] == "recommendation"
    monkeypatch.setattr(config, "UNKNOWN_GAP_THRESHOLD", gap + gap)
    assert classify(MIXED_TEXT) == (
        config.UNKNOWN_TYPE, config.UNKNOWN_CONFIDENCE
    )


@pytest.mark.parametrize(
    "text", ["", "12345", "Всё как обычно"],
    ids=["empty", "digits", "neutral_phrase"],
)
def test_text_without_signals_is_unknown(text: str) -> None:
    """Текст без признаков типа даёт unknown."""
    assert classify(text) == (config.UNKNOWN_TYPE, config.UNKNOWN_CONFIDENCE)


def test_weak_evidence_lowers_confidence(records: dict[str, str]) -> None:
    """Короткая R6 получает тип, но уверенность ниже порога."""
    _, weak = classify(records["R6"])
    _, strong = classify(records["R2"])
    assert weak < config.LOW_CONFIDENCE_THRESHOLD <= strong


def test_threshold_rule_selects_config_value(
    dataset: dict[str, Any], gold_records: dict[str, dict[str, Any]]
) -> None:
    """Правило подбора по данным R1–R6 даёт порог из конфига."""
    processed = [
        process_record(record, gold_records[record["id"]])
        for record in dataset["records"]
    ]
    selected = [row for row in threshold_sweep(processed)
                if row["selected_by_rule"]]
    assert [row["threshold"] for row in selected] == [
        config.UNKNOWN_GAP_THRESHOLD
    ]
