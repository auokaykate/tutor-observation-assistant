"""Тесты Части 1: извлечение полей из записей."""

from typing import Any

import pytest

from tutor_assistant import config
from tutor_assistant.extract import (
    FIELDS,
    extract,
    extract_author,
    extract_child_id,
    extract_date,
    extract_sleep_hours,
    extract_zone,
)

RECORD_IDS = ("R1", "R2", "R3", "R4", "R5", "R6")


@pytest.mark.parametrize(
    ("text", "iso_date"),
    [
        ("05.03.2025. Ребёнок CH-0421", "2025-03-05"),
        ("Отчёт о занятии №12 от 1 марта 2025 г. Специалист", "2025-03-01"),
        ("03/01/25, мама ребёнка CH-0342", "2025-03-01"),
        ("07.03.25 ch-0421 тьют.Иванова", "2025-03-07"),
        ("Рекомендация от 12 марта 2025 г. по ребёнку", "2025-03-12"),
        ("выгрузка от 2025-03-05", "2025-03-05"),
    ],
    ids=["dd.mm.yyyy", "text_month", "slash_mm_dd_yy", "dd.mm.yy",
         "text_month_two_digit_day", "iso"],
)
def test_date_formats(text: str, iso_date: str) -> None:
    """Каждый формат даты приводится к ISO."""
    assert extract_date(text) == iso_date


def test_slash_date_is_month_day_year() -> None:
    """03/01/25 читается как месяц/день/год, то есть 1 марта 2025."""
    assert extract_date("03/01/25") == "2025-03-01"


def test_slash_date_order_is_config_rule(monkeypatch: Any) -> None:
    """Порядок частей даты через «/» задаётся правилом в конфиге."""
    monkeypatch.setitem(
        config.DATE_FORMATS_BY_SEPARATOR, "/", ("%d/%m/%y",)
    )
    assert extract_date("03/01/25") == "2025-01-03"


@pytest.mark.parametrize(
    "text",
    [
        "31.02.2025 плакал",
        "Ребёнок был в хорошем настроении",
        "Отчёт о занятии №12",
        "Сон 6 ч 30 мин",
    ],
    ids=["impossible_date", "no_date", "number_sign", "duration_only"],
)
def test_date_missing_or_invalid(text: str) -> None:
    """Несуществующая или отсутствующая дата даёт None."""
    assert extract_date(text) is None


@pytest.mark.parametrize(
    ("text", "hours"),
    [
        ("Сон 6 ч 30 мин, утром вялый.", 6.5),
        ("Накануне спал 6,5 часов.", 6.5),
        ("ночью просыпался, итого 5:45 сна, утром", 5.75),
        ("сон~390 минут,настрй норм", 6.5),
        ("сон «6:30», утром бодрый", 6.5),
        ("заснул только к полуночи, спал 5 часов", 5.0),
        ("спал 7.5 ч", 7.5),
    ],
    ids=["h_min", "decimal_comma", "clock_5_45", "tilde_minutes",
         "quoted_clock_6_30", "whole_hours", "decimal_point"],
)
def test_sleep_formats(text: str, hours: float) -> None:
    """Каждый формат длительности сна переводится в часы."""
    assert extract_sleep_hours(text) == pytest.approx(hours)


@pytest.mark.parametrize(
    "text",
    [
        "плакал 20 минут без видимой причины",
        "поездка в центр на автобусе заняла 40 минут",
        "плакал 20 минут перед сном",
        "уснул в 22:30",
        "Сон хороший, занятие длилось 40 минут",
    ],
    ids=["crying", "trip", "before_sleep", "time_of_day", "comma_break"],
)
def test_sleep_not_taken_from_other_durations(text: str) -> None:
    """Длительности, не связанные со сном, не считаются сном."""
    assert extract_sleep_hours(text) is None


@pytest.mark.parametrize(
    ("text", "child_id"),
    [
        ("Ребёнок CH0107, зона: моторика.", "CH-0107"),
        ("07.03.25 ch-0421 тьют.Иванова", "CH-0421"),
        ("мама ребёнка CH-0342: ночью", "CH-0342"),
        ("ребёнок СН-0421, кириллица", "CH-0421"),
        ("по ребёнку Ch 0999", "CH-0999"),
    ],
    ids=["no_hyphen", "lower_case", "canonical", "cyrillic", "space"],
)
def test_child_id_normalization(text: str, child_id: str) -> None:
    """ID ребёнка приводится к виду CH-0421."""
    assert extract_child_id(text) == child_id


@pytest.mark.parametrize(
    "text",
    ["Ребёнок был в хорошем настроении", "CH-12345", "CH-042", "ARCH-0421"],
    ids=["absent", "five_digits", "three_digits", "inside_word"],
)
def test_child_id_missing(text: str) -> None:
    """Без корректного ID возвращается None."""
    assert extract_child_id(text) is None


@pytest.mark.parametrize(
    ("record_id", "author"),
    [
        ("R1", "Иванова А. С."),
        ("R2", "Петров И. И."),
        ("R3", "мама ребёнка"),
        ("R4", "Иванова"),
        ("R5", "Петров И. И."),
        ("R6", None),
    ],
    ids=RECORD_IDS,
)
def test_author(
    records: dict[str, str], record_id: str, author: str | None
) -> None:
    """Автор определяется по роли сотрудника или шапке записи родителя."""
    assert extract_author(records[record_id]) == author


def test_parent_mentioned_in_text_is_not_author() -> None:
    """Упоминание мамы в середине текста не делает её автором."""
    assert extract_author("Ребёнок сказал, что мама купила пазл") is None


def test_explicit_zone_is_extracted(records: dict[str, str]) -> None:
    """В R2 зона указана явно: «зона: моторика»."""
    assert extract(records["R2"])["zone"] == "моторика"


def test_zone_is_not_inferred_from_r4(records: dict[str, str]) -> None:
    """«сенсор.перегрузка» в R4 не явная зона, поэтому zone равно None."""
    assert extract(records["R4"])["zone"] is None


def test_explicit_zone_must_be_tracked() -> None:
    """Явная пометка с зоной не из списка не извлекается."""
    assert extract_zone("зона: игра") is None
    assert extract_zone("Зона — Сон") == "сон"


def test_r6_has_no_fields(records: dict[str, str]) -> None:
    """В R6 нет ни одного поля, все значения None."""
    result = extract(records["R6"])
    assert tuple(result) == FIELDS
    assert all(value is None for value in result.values())


@pytest.mark.parametrize("record_id", RECORD_IDS)
def test_records_match_expected(
    records: dict[str, str],
    gold_records: dict[str, dict[str, Any]],
    record_id: str,
) -> None:
    """Все поля R1–R6 совпадают с ручной разметкой."""
    gold = gold_records[record_id]
    assert extract(records[record_id]) == {
        field: gold[field] for field in FIELDS
    }
