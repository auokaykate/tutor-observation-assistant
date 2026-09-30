"""Часть 1: детерминированное извлечение полей из текстовой записи.

Функция extract возвращает словарь с ключами date, child_id, author,
sleep_hours и zone. Всё, что не найдено в тексте явно, возвращается как
None: значения не достраиваются по смыслу и не берутся из других записей.
"""

import re
from collections.abc import Callable
from datetime import date, datetime

from tutor_assistant import config
from tutor_assistant.text import collapse_spaces, normalize

FIELDS: tuple[str, ...] = ("date", "child_id", "author", "sleep_hours", "zone")

ExtractedValue = str | float | None


def _alternation(words: tuple[str, ...]) -> str:
    """Собрать альтернативу для регулярного выражения, длинные слова первыми.

    Args:
        words: Слова или фразы.

    Returns:
        Строка вида "часов|часа|час|ч" с экранированными элементами.
    """
    ordered = sorted(words, key=len, reverse=True)
    return "|".join(re.escape(word) for word in ordered)


_NUMERIC_DATE_RE = re.compile(
    r"(?<![\d.,/-])\d{1,4}(?P<sep>[./-])\d{1,2}(?P=sep)\d{1,4}"
    r"(?![\d/-]|[.,]\d)"
)
_TEXT_DATE_RE = re.compile(
    rf"(?<!\d)(\d{{1,2}})\s+({_alternation(tuple(config.MONTHS_GENITIVE))})"
    r"\s+(\d{4})(?!\d)",
    re.IGNORECASE,
)

_CHILD_ID_BODY = (
    f"{config.CHILD_ID_PREFIX_PATTERN}{config.CHILD_ID_SEPARATOR_PATTERN}"
    rf"\d{{{config.CHILD_ID_DIGITS}}}"
)
_CHILD_ID_RE = re.compile(
    rf"(?<![A-Za-zА-Яа-яЁё]){config.CHILD_ID_PREFIX_PATTERN}"
    rf"{config.CHILD_ID_SEPARATOR_PATTERN}"
    rf"(\d{{{config.CHILD_ID_DIGITS}}})(?!\d)"
)

_STAFF_AUTHOR_RE = re.compile(
    rf"(?i:\b(?:{'|'.join(config.STAFF_ROLE_PATTERNS)}))\s*"
    r"(?P<name>[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
    r"(?:\s+[А-ЯЁ]\.(?:\s?[А-ЯЁ]\.)?)?)"
)
_PARENT_AUTHOR_RE = re.compile(
    rf"(?i:\b(?P<author>(?:{_alternation(config.PARENT_AUTHOR_WORDS)})"
    rf"{config.PARENT_AUTHOR_SUFFIX_PATTERN}))"
    rf"(?=\s*(?:{_CHILD_ID_BODY})?\s*:)"
)

_HOURS = _alternation(config.HOUR_UNITS)
_MINUTES = _alternation(config.MINUTE_UNITS)
_WORD_END = r"(?![а-яё])"
_HOURS_MINUTES_RE = re.compile(
    rf"(?<![\d.,])(?P<h>\d{{1,2}})\s*(?:{_HOURS})\.?\s*"
    rf"(?P<m>\d{{1,2}})\s*(?:{_MINUTES}){_WORD_END}",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"(?<![\d:.,])(?P<h>\d{1,2}):(?P<m>[0-5]\d)(?![\d:])")
_HOURS_RE = re.compile(
    rf"(?<![\d.,])(?P<h>\d{{1,2}}(?:[.,]\d{{1,2}})?)\s*"
    rf"(?:{_HOURS}){_WORD_END}",
    re.IGNORECASE,
)
_MINUTES_RE = re.compile(
    rf"(?<![\d.,])(?P<m>\d{{1,4}})\s*(?:{_MINUTES}){_WORD_END}",
    re.IGNORECASE,
)
_SLEEP_WORD_RE = re.compile(
    rf"(?<![а-яё])(?:{_alternation(config.SLEEP_WORDS)}){_WORD_END}",
    re.IGNORECASE,
)
_SLEEP_PREPOSITION_RE = re.compile(
    rf"(?<![а-яё])(?:{_alternation(config.SLEEP_TIME_PREPOSITIONS)})\s*$",
    re.IGNORECASE,
)
_CLOCK_PREPOSITION_RE = re.compile(
    rf"(?<![а-яё])(?:{_alternation(config.CLOCK_TIME_PREPOSITIONS)})\s*$",
    re.IGNORECASE,
)
_EXPLICIT_ZONE_RE = re.compile(config.EXPLICIT_ZONE_PATTERN, re.IGNORECASE)


def _parse_numeric_date(match: re.Match[str]) -> date | None:
    """Разобрать числовую дату по форматам для её разделителя из конфига.

    Args:
        match: Совпадение с числовой датой.

    Returns:
        Дата или None, если ни один формат не подошёл.
    """
    formats = config.DATE_FORMATS_BY_SEPARATOR.get(match.group("sep"), ())
    for date_format in formats:
        try:
            return datetime.strptime(match.group(0), date_format).date()
        except ValueError:
            continue
    return None


def _parse_text_date(match: re.Match[str]) -> date | None:
    """Разобрать дату вида «1 марта 2025».

    Args:
        match: Совпадение с датой, где месяц записан словом.

    Returns:
        Дата или None, если такой даты не существует.
    """
    day, month_name, year = match.groups()
    month = config.MONTHS_GENITIVE[month_name.lower()]
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def extract_date(text: str) -> str | None:
    """Найти первую корректную дату в тексте.

    Поддерживаются 05.03.2025, 07.03.25, 03/01/25 (месяц/день/год, правило
    в config.DATE_FORMATS_BY_SEPARATOR), 2025-03-05 и «1 марта 2025 г.».

    Args:
        text: Текст записи.

    Returns:
        Дата в формате ISO (config.ISO_DATE_FORMAT) или None.
    """
    parsers: list[tuple[int, Callable[[re.Match[str]], date | None],
                        re.Match[str]]] = []
    parsers.extend(
        (m.start(), _parse_numeric_date, m)
        for m in _NUMERIC_DATE_RE.finditer(text)
    )
    parsers.extend(
        (m.start(), _parse_text_date, m) for m in _TEXT_DATE_RE.finditer(text)
    )
    for _, parser, match in sorted(parsers, key=lambda item: item[0]):
        parsed = parser(match)
        if parsed is not None:
            return parsed.strftime(config.ISO_DATE_FORMAT)
    return None


def extract_child_id(text: str) -> str | None:
    """Найти ID ребёнка и привести его к виду CH-0421.

    Принимаются варианты CH-0421, CH0107, ch-0421, а также «СН» кириллицей.

    Args:
        text: Текст записи.

    Returns:
        Нормализованный ID или None.
    """
    match = _CHILD_ID_RE.search(text)
    if match is None:
        return None
    return config.CHILD_ID_TEMPLATE.format(digits=match.group(1))


def extract_author(text: str) -> str | None:
    """Найти автора записи.

    Автор — тот, кто сделал запись. Сотрудник определяется по роли перед
    фамилией («тьютор Иванова А. С.», «тьют.Иванова», «Специалист Петров
    И. И.»), возвращается фамилия с инициалами без роли. Родитель
    определяется по шапке записи («мама ребёнка CH-0342: ...») и
    возвращается как написано: имени в тексте нет, а роль автора важна.

    Args:
        text: Текст записи.

    Returns:
        Автор или None.
    """
    staff = _STAFF_AUTHOR_RE.search(text)
    if staff is not None:
        return collapse_spaces(staff.group("name"))
    parent = _PARENT_AUTHOR_RE.search(text)
    if parent is not None:
        return collapse_spaces(parent.group("author"))
    return None


def _duration_candidates(text: str) -> list[tuple[int, int, float]]:
    """Найти все длительности в тексте без учёта контекста сна.

    Шаблоны применяются от более специфичного к общему, пересекающиеся
    совпадения отбрасываются.

    Args:
        text: Текст записи.

    Returns:
        Список (начало, конец, длительность в часах).
    """
    found: list[tuple[int, int, float]] = []

    def overlaps(start: int, end: int) -> bool:
        """Проверить пересечение с уже найденными длительностями."""
        return any(start < f_end and f_start < end
                   for f_start, f_end, _ in found)

    for match in _HOURS_MINUTES_RE.finditer(text):
        minutes = int(match.group("m"))
        if minutes < config.MINUTES_PER_HOUR:
            hours = int(match.group("h")) + minutes / config.MINUTES_PER_HOUR
            found.append((match.start(), match.end(), hours))
    for match in _CLOCK_RE.finditer(text):
        if overlaps(match.start(), match.end()):
            continue
        if _CLOCK_PREPOSITION_RE.search(text[:match.start()]):
            continue
        hours = (int(match.group("h"))
                 + int(match.group("m")) / config.MINUTES_PER_HOUR)
        found.append((match.start(), match.end(), hours))
    for match in _HOURS_RE.finditer(text):
        if not overlaps(match.start(), match.end()):
            value = match.group("h").replace(
                config.DECIMAL_COMMA, config.DECIMAL_POINT
            )
            found.append((match.start(), match.end(), float(value)))
    for match in _MINUTES_RE.finditer(text):
        if not overlaps(match.start(), match.end()):
            hours = int(match.group("m")) / config.MINUTES_PER_HOUR
            found.append((match.start(), match.end(), hours))
    return found


def _sleep_gap(text: str, word: re.Match[str], start: int,
               end: int) -> int | None:
    """Посчитать расстояние между словом сна и длительностью.

    Args:
        text: Текст записи.
        word: Совпадение со словом сна.
        start: Начало длительности.
        end: Конец длительности.

    Returns:
        Число символов между ними или None, если связи нет: слишком
        далеко или между ними стоит разрывающий знак препинания.
    """
    if word.end() <= start:
        between = text[word.end():start]
    elif word.start() >= end:
        between = text[end:word.start()]
    else:
        return None
    if len(between) > config.SLEEP_CONTEXT_MAX_GAP:
        return None
    if any(char in config.SLEEP_CONTEXT_BREAKERS for char in between):
        return None
    return len(between)


def extract_sleep_hours(text: str) -> float | None:
    """Найти длительность сна в часах.

    Поддерживаются «6 ч 30 мин», «6,5 часов», «5:45», «390 минут», «6:30».
    Длительность засчитывается как сон, только если рядом стоит слово сна
    («сон», «сна», «спал» и т. п.) без разрывающих знаков между ними.

    Args:
        text: Текст записи.

    Returns:
        Часы сна, округлённые до config.SLEEP_HOURS_DIGITS, или None.
    """
    words = [
        word for word in _SLEEP_WORD_RE.finditer(text)
        if not _SLEEP_PREPOSITION_RE.search(text[:word.start()])
    ]
    best: tuple[int, int, float] | None = None
    for start, end, hours in _duration_candidates(text):
        if not 0 < hours <= config.MAX_SLEEP_HOURS:
            continue
        gaps = [_sleep_gap(text, word, start, end) for word in words]
        valid = [gap for gap in gaps if gap is not None]
        if not valid:
            continue
        candidate = (min(valid), start, hours)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    return round(best[2], config.SLEEP_HOURS_DIGITS)


def extract_zone(text: str) -> str | None:
    """Найти зону, только если она указана явно («зона: моторика»).

    Зона по смыслу («сенсор.перегрузка») здесь не определяется — это
    задача check_zone из Части 3.

    Args:
        text: Текст записи.

    Returns:
        Зона из config.ZONES или None.
    """
    for match in _EXPLICIT_ZONE_RE.finditer(text):
        value = normalize(match.group("zone"))
        for zone in config.ZONES:
            if normalize(zone) == value:
                return zone
    return None


def extract(text: str) -> dict[str, ExtractedValue]:
    """Извлечь поля date, child_id, author, sleep_hours и zone из записи.

    Args:
        text: Текст записи в свободной форме.

    Returns:
        Словарь с ключами из FIELDS; ненайденные поля равны None.
    """
    clean = collapse_spaces(text)
    return {
        "date": extract_date(clean),
        "child_id": extract_child_id(clean),
        "author": extract_author(clean),
        "sleep_hours": extract_sleep_hours(clean),
        "zone": extract_zone(clean),
    }
