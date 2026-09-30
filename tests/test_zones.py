"""Тесты Части 3: проверка зон на правилах (fallback без ключа и сети)."""

from typing import Any

import pytest

from tutor_assistant import config
from tutor_assistant.text import extract_numbers
from tutor_assistant.zones import check_zone, check_zone_detailed

OBSERVATION_NUMBERS = tuple(range(1, 16))


def _case(
    dataset: dict[str, Any], expected: dict[str, Any], number: int
) -> tuple[str, dict[str, Any]]:
    """Вернуть наблюдение и его разметку по номеру.

    Args:
        dataset: Содержимое dataset.json.
        expected: Содержимое expected.json.
        number: Номер наблюдения, начиная с 1.

    Returns:
        Текст наблюдения и ожидаемый ответ.
    """
    return (
        dataset["observations"][number - 1],
        expected["observations"][number - 1],
    )


@pytest.mark.parametrize("number", OBSERVATION_NUMBERS)
def test_observation_matches_expected(
    dataset: dict[str, Any], expected: dict[str, Any], number: int
) -> None:
    """Каждое из наблюдений на fallback совпадает с разметкой."""
    note, gold = _case(dataset, expected, number)
    result = check_zone_detailed(note)
    assert result.source == config.SOURCE_RULES
    assert result.is_zone == gold["is_zone"]
    assert 0.0 <= result.confidence <= 1.0
    assert result.explanation
    if gold["is_zone"]:
        assert result.zone in gold["zones"]
        assert f"'{result.zone}'" in result.explanation
    else:
        assert result.zone is None
        assert result.explanation.startswith("Не относится")


@pytest.mark.parametrize("number", OBSERVATION_NUMBERS)
def test_disputed_cases_have_lower_confidence(
    dataset: dict[str, Any], expected: dict[str, Any], number: int
) -> None:
    """Спорные случаи ниже порога и с сомнением, однозначные — выше."""
    note, gold = _case(dataset, expected, number)
    result = check_zone_detailed(note)
    if gold["disputed"]:
        assert result.confidence < config.LOW_CONFIDENCE_THRESHOLD
        assert "Сомнение" in result.explanation
        assert result.doubts
    else:
        assert result.confidence >= config.LOW_CONFIDENCE_THRESHOLD
        assert "Сомнение" not in result.explanation


@pytest.mark.parametrize("number", OBSERVATION_NUMBERS)
def test_explanation_has_no_invented_numbers(
    dataset: dict[str, Any], number: int
) -> None:
    """В объяснении нет чисел, которых нет в тексте наблюдения."""
    note = dataset["observations"][number - 1]
    _, _, explanation = check_zone(note)
    assert extract_numbers(explanation) <= extract_numbers(note)


def test_food_refusal_example_from_task() -> None:
    """Пример из задания: отказ от еды относится к зоне 'питание'."""
    is_zone, confidence, explanation = check_zone(
        "отказ от еды третий день подряд"
    )
    assert is_zone is True
    assert confidence >= config.LOW_CONFIDENCE_THRESHOLD
    assert explanation.startswith("Отказ от еды относится к зоне 'питание'")


def test_numbers_from_text_are_kept() -> None:
    """Числа из наблюдения попадают в объяснение без искажений."""
    _, _, explanation = check_zone("заснул только к полуночи, спал 5 часов")
    assert "спал 5 часов" in explanation


def test_note_without_signals_needs_clarification() -> None:
    """Без признаков зон и обстоятельств — False с низкой уверенностью."""
    result = check_zone_detailed("всё как обычно")
    assert result.is_zone is False
    assert result.confidence == config.NO_SIGNAL_CONFIDENCE
    assert result.confidence < config.LOW_CONFIDENCE_THRESHOLD


def test_noise_making_is_not_sensory() -> None:
    """«шумел» — поведение ребёнка, а не реакция на шум."""
    result = check_zone_detailed("на перемене шумел и кричал")
    assert result.zone == "поведение"
    assert result.scores["сенсорика"] == 0


def test_food_refusal_is_not_behaviour() -> None:
    """Отказ от еды не засчитывается как отказ от деятельности."""
    result = check_zone_detailed("отказ от еды третий день подряд")
    assert result.scores["поведение"] == 0
    assert not result.doubts


def test_any_doubt_goes_below_threshold() -> None:
    """Одно сомнение гарантированно опускает уверенность ниже порога."""
    assert (
        config.ZONE_MAX_CONFIDENCE * config.ZONE_DOUBT_FACTOR
        < config.LOW_CONFIDENCE_THRESHOLD
    )


def test_check_zone_returns_tuple() -> None:
    """check_zone возвращает (bool, float, str)."""
    result = check_zone("плакал 20 минут без видимой причины")
    assert isinstance(result, tuple)
    is_zone, confidence, explanation = result
    assert isinstance(is_zone, bool)
    assert isinstance(confidence, float)
    assert isinstance(explanation, str)
