"""Часть 3: проверка отнесения наблюдения к отслеживаемым зонам развития.

Основной путь без сети — правила из config.ZONE_RULES. LLM-путь включается
явно и при любой проблеме молча уступает правилам. Спорность не
угадывается: конкурирующая зона или ситуативное обстоятельство снижают
уверенность ниже config.LOW_CONFIDENCE_THRESHOLD и описываются в
объяснении.
"""

import re
from dataclasses import dataclass
from typing import Any

from tutor_assistant import config, llm
from tutor_assistant.text import clause_at, collapse_spaces, normalize

_RULES: tuple[tuple[config.ZoneRule, re.Pattern[str]], ...] = tuple(
    sorted(
        ((rule, re.compile(rule.pattern)) for rule in config.ZONE_RULES),
        key=lambda item: -item[0].weight,
    )
)
_ORGANIZATIONAL: dict[str, re.Pattern[str]] = {
    category: re.compile(pattern)
    for category, pattern in config.ORGANIZATIONAL_MARKERS.items()
}
_SITUATIONAL: dict[str, re.Pattern[str]] = {
    category: re.compile(pattern)
    for category, pattern in config.SITUATIONAL_MARKERS.items()
}


@dataclass(frozen=True)
class Evidence:
    """Сработавший признак зоны.

    Attributes:
        zone: Зона признака.
        label: Название признака.
        weight: Вес признака.
        start: Начало совпадения в тексте.
        end: Конец совпадения в тексте.
    """

    zone: str
    label: str
    weight: float
    start: int
    end: int


@dataclass(frozen=True)
class Marker:
    """Найденное в тексте обстоятельство.

    Attributes:
        category: Категория обстоятельства.
        spans: Позиции всех совпадений категории в тексте.
    """

    category: str
    spans: tuple[tuple[int, int], ...]

    def fragments(self, text: str) -> str:
        """Вернуть совпавшие слова через запятую.

        Args:
            text: Исходный текст наблюдения.

        Returns:
            Строка из совпавших фрагментов текста.
        """
        return ", ".join(text[start:end] for start, end in self.spans)


@dataclass(frozen=True)
class ZoneResult:
    """Подробный результат проверки зоны.

    Attributes:
        is_zone: Относится ли наблюдение к отслеживаемым зонам.
        confidence: Уверенность в ответе от 0.0 до 1.0.
        explanation: Объяснение с зоной и причиной.
        zone: Основная зона или None.
        source: Источник ответа: config.SOURCE_RULES или config.SOURCE_LLM.
        scores: Скоры зон по правилам (пусто для ответа LLM).
        doubts: Краткие причины сомнений.
    """

    is_zone: bool
    confidence: float
    explanation: str
    zone: str | None
    source: str
    scores: dict[str, float]
    doubts: tuple[str, ...]

    def as_tuple(self) -> tuple[bool, float, str]:
        """Вернуть результат в формате функции check_zone.

        Returns:
            Тройка (относится ли к зонам, уверенность, объяснение).
        """
        return self.is_zone, self.confidence, self.explanation


def collect_evidence(normalized: str) -> dict[str, list[Evidence]]:
    """Найти признаки зон в нормализованном тексте.

    Правила применяются от большего веса к меньшему. Внутри одной зоны
    пересекающиеся совпадения не суммируются, каждое правило учитывается
    один раз.

    Args:
        normalized: Текст после normalize.

    Returns:
        Признаки по зонам в порядке убывания веса.
    """
    evidence: dict[str, list[Evidence]] = {zone: [] for zone in config.ZONES}
    for rule, pattern in _RULES:
        accepted = evidence[rule.zone]
        for match in pattern.finditer(normalized):
            overlapping = any(
                match.start() < item.end and item.start < match.end()
                for item in accepted
            )
            if not overlapping:
                accepted.append(
                    Evidence(
                        rule.zone, rule.label, rule.weight,
                        match.start(), match.end(),
                    )
                )
                break
    return evidence


def find_markers(
    normalized: str, markers: dict[str, re.Pattern[str]]
) -> list[Marker]:
    """Найти обстоятельства заданных категорий.

    Args:
        normalized: Текст после normalize.
        markers: Скомпилированные шаблоны по категориям.

    Returns:
        Найденные категории со всеми совпадениями.
    """
    found: list[Marker] = []
    for category, pattern in markers.items():
        spans = tuple(match.span() for match in pattern.finditer(normalized))
        if spans:
            found.append(Marker(category, spans))
    return found


def _quote_clauses(text: str, spans: list[tuple[int, int]]) -> str:
    """Процитировать фрагменты текста, содержащие совпадения.

    Args:
        text: Исходный текст наблюдения.
        spans: Позиции совпадений.

    Returns:
        Фрагменты в кавычках через запятую, без повторов.
    """
    clauses: list[str] = []
    for start, end in sorted(spans):
        clause = clause_at(text, start, end)
        if clause and clause not in clauses:
            clauses.append(clause)
    return ", ".join(f"«{clause}»" for clause in clauses)


def _lower_first(phrase: str) -> str:
    """Сделать первую букву строчной.

    Args:
        phrase: Фраза.

    Returns:
        Фраза со строчной первой буквой.
    """
    return phrase[:1].lower() + phrase[1:]


def _evidence_confidence(score: float) -> float:
    """Перевести скор зоны в уверенность с насыщением.

    Args:
        score: Скор лучшей зоны.

    Returns:
        config.ZONE_MAX_CONFIDENCE * s / (s + половинное насыщение).
    """
    return (
        config.ZONE_MAX_CONFIDENCE * score
        / (score + config.ZONE_EVIDENCE_HALF_SATURATION)
    )


def _non_zone_result(
    text: str, markers: list[Marker], scores: dict[str, float]
) -> ZoneResult:
    """Собрать ответ False для наблюдения без признаков зон.

    Args:
        text: Исходный текст наблюдения.
        markers: Найденные обстоятельства.
        scores: Нулевые скоры зон.

    Returns:
        Результат с is_zone=False и объяснением причины.
    """
    if markers:
        details = "; ".join(
            f"{marker.category} ({_quote_clauses(text, list(marker.spans))})"
            for marker in markers
        )
        explanation = (
            "Не относится ни к одной зоне: описано обстоятельство — "
            f"{details}, а не состояние или действия ребёнка. Сомнение: "
            "реакция ребёнка не описана; если она была, запись нужно "
            "уточнить."
        )
        return ZoneResult(
            is_zone=False,
            confidence=config.NON_ZONE_CONFIDENCE,
            explanation=explanation,
            zone=None,
            source=config.SOURCE_RULES,
            scores=scores,
            doubts=("реакция ребёнка не описана",),
        )
    explanation = (
        "Признаков отслеживаемых зон и знакомых обстоятельств не найдено: "
        "словарь правил ограничен, поэтому вывод неуверенный и запись "
        "нужно проверить."
    )
    return ZoneResult(
        is_zone=False,
        confidence=config.NO_SIGNAL_CONFIDENCE,
        explanation=explanation,
        zone=None,
        source=config.SOURCE_RULES,
        scores=scores,
        doubts=("признаки не распознаны",),
    )


def check_zone_rules(note: str) -> ZoneResult:
    """Проверить зону наблюдения только по правилам, без сети.

    Уверенность растёт со скором лучшей зоны и умножается на
    config.ZONE_DOUBT_FACTOR за каждый вид сомнения: конкурирующая зона
    и ситуативное обстоятельство (погода, праздник).

    Args:
        note: Текст наблюдения.

    Returns:
        Подробный результат с источником config.SOURCE_RULES.
    """
    text = collapse_spaces(note)
    normalized = normalize(text)
    evidence = collect_evidence(normalized)
    scores = {
        zone: sum(item.weight for item in items)
        for zone, items in evidence.items()
    }
    organizational = find_markers(normalized, _ORGANIZATIONAL)
    situational = find_markers(normalized, _SITUATIONAL)
    ranking = sorted(
        config.ZONES,
        key=lambda zone: (-scores[zone], config.ZONES.index(zone)),
    )
    top = ranking[0]
    if not scores[top]:
        return _non_zone_result(text, organizational + situational, scores)

    top_items = evidence[top]
    sentences = [
        f"{top_items[0].label} относится к зоне '{top}' (в записи: "
        f"{_quote_clauses(text, [(i.start, i.end) for i in top_items])})."
    ]
    confidence = _evidence_confidence(scores[top])
    doubts: list[str] = []

    competing = [
        zone for zone in ranking[1:]
        if scores[zone]
        and scores[zone] >= config.COMPETING_ZONE_RATIO * scores[top]
    ]
    if competing:
        confidence *= config.ZONE_DOUBT_FACTOR
        for zone in competing:
            items = evidence[zone]
            quotes = _quote_clauses(text, [(i.start, i.end) for i in items])
            sentences.append(
                f"Сомнение: есть признаки и зоны '{zone}' "
                f"({_lower_first(items[0].label)}: {quotes})."
            )
            doubts.append(f"конкурирующая зона '{zone}'")
    if situational:
        confidence *= config.ZONE_DOUBT_FACTOR
        for marker in situational:
            sentences.append(
                f"Сомнение: названо обстоятельство ({marker.category}: "
                f"«{marker.fragments(text)}»), реакция может быть "
                "ситуативной."
            )
            doubts.append(f"обстоятельство: {marker.category}")
    for marker in organizational:
        sentences.append(
            f"Контекст: {marker.category} («{marker.fragments(text)}»)."
        )
    if doubts:
        sentences.append("Уверенность снижена, нужна проверка специалистом.")

    return ZoneResult(
        is_zone=True,
        confidence=round(confidence, config.CONFIDENCE_DIGITS),
        explanation=" ".join(sentences),
        zone=top,
        source=config.SOURCE_RULES,
        scores=scores,
        doubts=tuple(doubts),
    )


def check_zone_detailed(
    note: str, *, use_llm: bool | None = None, client: Any | None = None
) -> ZoneResult:
    """Проверить зону наблюдения: LLM при включении, иначе правила.

    Args:
        note: Текст наблюдения.
        use_llm: Пробовать ли LLM; по умолчанию config.USE_LLM_BY_DEFAULT.
        client: Клиент Anthropic или заглушка для тестов.

    Returns:
        Подробный результат с указанием источника.
    """
    enabled = config.USE_LLM_BY_DEFAULT if use_llm is None else use_llm
    if enabled:
        answer = llm.ask_llm(note, client=client)
        if answer is not None:
            return ZoneResult(
                is_zone=answer.is_zone,
                confidence=answer.confidence,
                explanation=answer.explanation,
                zone=answer.zone,
                source=config.SOURCE_LLM,
                scores={},
                doubts=(),
            )
    return check_zone_rules(note)


def check_zone(
    note: str, *, use_llm: bool | None = None, client: Any | None = None
) -> tuple[bool, float, str]:
    """Определить, относится ли наблюдение к отслеживаемым зонам.

    Args:
        note: Текст наблюдения.
        use_llm: Пробовать ли LLM перед правилами.
        client: Клиент Anthropic или заглушка для тестов.

    Returns:
        Тройка (относится ли к зонам, уверенность, объяснение), например
        (True, 0.76, "Отказ от еды относится к зоне 'питание' ...").
    """
    return check_zone_detailed(note, use_llm=use_llm, client=client).as_tuple()
