"""Общие функции работы с текстом: нормализация, фрагменты, числа."""

import re

from tutor_assistant import config

_NUMBER_RE = re.compile(config.NUMBER_PATTERN)
_TRANSLATION = str.maketrans(config.NORMALIZATION_MAP)


def normalize(text: str) -> str:
    """Привести текст к нижнему регистру и заменить «ё» на «е».

    Длина строки не меняется, поэтому позиции совпадений в нормализованном
    тексте совпадают с позициями в исходном.

    Args:
        text: Исходный текст.

    Returns:
        Нормализованный текст той же длины.
    """
    lowered = "".join(
        char.lower() if len(char.lower()) == len(char) else char
        for char in text
    )
    return lowered.translate(_TRANSLATION)


def collapse_spaces(text: str) -> str:
    """Заменить последовательности пробельных символов одним пробелом.

    Args:
        text: Исходный текст.

    Returns:
        Текст без переносов строк и повторных пробелов.
    """
    return " ".join(text.split())


def clause_at(text: str, start: int, end: int) -> str:
    """Вернуть фрагмент текста между разделителями вокруг совпадения.

    Args:
        text: Исходный текст.
        start: Начало совпадения.
        end: Конец совпадения.

    Returns:
        Фрагмент без крайних пробелов, ограниченный знаками препинания
        из config.CLAUSE_SEPARATORS или краями текста.
    """
    left = max(text.rfind(sep, 0, start) for sep in config.CLAUSE_SEPARATORS)
    positions = (text.find(sep, end) for sep in config.CLAUSE_SEPARATORS)
    right = min((pos for pos in positions if pos != -1), default=len(text))
    return text[left + 1:right].strip()


def extract_numbers(text: str) -> set[str]:
    """Вернуть числа из текста, приведённые к записи с точкой.

    Args:
        text: Текст для поиска чисел.

    Returns:
        Множество строковых представлений чисел, например {"5", "6.5"}.
    """
    return {
        number.replace(config.DECIMAL_COMMA, config.DECIMAL_POINT)
        for number in _NUMBER_RE.findall(text)
    }
