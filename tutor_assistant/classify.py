"""Часть 2: классификация типа записи эвристиками по ключевым словам.

Для каждого типа считается скор — сумма весов сработавших признаков из
config.CLASSIFY_KEYWORDS. Скоры переводятся в доли от суммы. Если разрыв
долей первого и второго места меньше порога, возвращается unknown.
"""

import re
from dataclasses import dataclass

from tutor_assistant import config
from tutor_assistant.text import normalize

_KEYWORDS: dict[str, tuple[tuple[re.Pattern[str], float], ...]] = {
    record_type: tuple(
        (re.compile(pattern), weight) for pattern, weight in rules.items()
    )
    for record_type, rules in config.CLASSIFY_KEYWORDS.items()
}


@dataclass(frozen=True)
class ClassificationResult:
    """Подробный результат классификации одной записи.

    Attributes:
        label: Выбранный тип или unknown.
        confidence: Уверенность от 0.0 до 1.0.
        scores: Сырые скоры по типам.
        shares: Доли скоров от суммы.
        leader: Тип с наибольшей долей или None, если признаков нет.
        runner_up: Тип на втором месте или None.
        gap: Разрыв долей первого и второго места.
        evidence: Фрагменты текста, давшие вклад в скор каждого типа.
    """

    label: str
    confidence: float
    scores: dict[str, float]
    shares: dict[str, float]
    leader: str | None
    runner_up: str | None
    gap: float
    evidence: dict[str, list[str]]

    def as_tuple(self) -> tuple[str, float]:
        """Вернуть результат в формате функции classify.

        Returns:
            Пара (тип записи, уверенность).
        """
        return self.label, self.confidence


def score_types(text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Посчитать скоры типов записи по ключевым словам.

    Каждый признак учитывается один раз, даже если встречается несколько
    раз.

    Args:
        text: Текст записи.

    Returns:
        Скоры по типам и фрагменты исходного текста, которые их дали.
    """
    normalized = normalize(text)
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for record_type in config.RECORD_TYPES:
        scores[record_type] = 0.0
        evidence[record_type] = []
        for pattern, weight in _KEYWORDS[record_type]:
            match = pattern.search(normalized)
            if match is not None:
                scores[record_type] += weight
                evidence[record_type].append(text[match.start():match.end()])
    return scores, evidence


def classify_detailed(
    text: str, threshold: float | None = None
) -> ClassificationResult:
    """Классифицировать запись и вернуть все промежуточные величины.

    Args:
        text: Текст записи.
        threshold: Порог разрыва для unknown; по умолчанию
            config.UNKNOWN_GAP_THRESHOLD.

    Returns:
        Подробный результат классификации.
    """
    gap_threshold = (
        config.UNKNOWN_GAP_THRESHOLD if threshold is None else threshold
    )
    scores, evidence = score_types(text)
    total = sum(scores.values())
    if not total:
        return ClassificationResult(
            label=config.UNKNOWN_TYPE,
            confidence=config.UNKNOWN_CONFIDENCE,
            scores=scores,
            shares={record_type: 0.0 for record_type in config.RECORD_TYPES},
            leader=None,
            runner_up=None,
            gap=0.0,
            evidence=evidence,
        )
    shares = {
        record_type: round(score / total, config.SCORE_DIGITS)
        for record_type, score in scores.items()
    }
    ranking = sorted(
        config.RECORD_TYPES,
        key=lambda record_type: (
            -shares[record_type],
            config.RECORD_TYPES.index(record_type),
        ),
    )
    leader, runner_up = ranking[0], ranking[1]
    gap = round(shares[leader] - shares[runner_up], config.SCORE_DIGITS)
    if gap < gap_threshold:
        label = config.UNKNOWN_TYPE
        confidence = config.UNKNOWN_CONFIDENCE
    else:
        label = leader
        evidence_factor = min(1.0, total / config.CLASSIFY_EVIDENCE_SATURATION)
        weighted_share = shares[leader] * evidence_factor
        confidence = round(
            min(config.CLASSIFY_MAX_CONFIDENCE, weighted_share),
            config.CONFIDENCE_DIGITS,
        )
    return ClassificationResult(
        label=label,
        confidence=confidence,
        scores=scores,
        shares=shares,
        leader=leader,
        runner_up=runner_up,
        gap=gap,
        evidence=evidence,
    )


def classify(text: str) -> tuple[str, float]:
    """Определить тип записи и уверенность.

    Типы: observation, lesson_report, parent_note, recommendation, unknown.
    Уверенность — доля скора лидера, умноженная на насыщение по объёму
    улик (сумма весов / config.CLASSIFY_EVIDENCE_SATURATION, не больше 1),
    и ограниченная config.CLASSIFY_MAX_CONFIDENCE. Для unknown уверенность
    равна config.UNKNOWN_CONFIDENCE: тип не выбран.

    Args:
        text: Текст записи.

    Returns:
        Пара (тип записи, уверенность от 0.0 до 1.0).
    """
    return classify_detailed(text).as_tuple()
