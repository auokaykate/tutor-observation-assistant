"""Необязательный LLM-путь для check_zone через Anthropic SDK.

Модуль ничего не бросает наружу: при отсутствии пакета anthropic, ключа
или сети, при неверном ключе, отказе модели или ответе не того формата
функция ask_llm возвращает None, и check_zone молча переходит на правила.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from tutor_assistant import config
from tutor_assistant.text import extract_numbers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMAnswer:
    """Ответ модели, прошедший проверку формата.

    Attributes:
        is_zone: Относится ли наблюдение к отслеживаемым зонам.
        zone: Зона из config.ZONES или None.
        confidence: Уверенность от 0.0 до 1.0.
        explanation: Объяснение с названием зоны и причиной.
    """

    is_zone: bool
    zone: str | None
    confidence: float
    explanation: str


def response_schema() -> dict[str, Any]:
    """Вернуть JSON-схему ответа для structured outputs.

    Returns:
        Схема объекта с полями related, zone, confidence, explanation.
    """
    return {
        "type": "object",
        "properties": {
            "related": {"type": "boolean"},
            "zone": {
                "type": "string",
                "enum": [*config.ZONES, config.LLM_NO_ZONE_LABEL],
            },
            "confidence": {"type": "number"},
            "explanation": {"type": "string"},
        },
        "required": list(config.LLM_ANSWER_KEYS),
        "additionalProperties": False,
    }


def system_prompt() -> str:
    """Собрать системный промпт со списком зон.

    Returns:
        Текст системного промпта.
    """
    return config.LLM_SYSTEM_PROMPT.format(
        zones=", ".join(config.ZONES), no_zone=config.LLM_NO_ZONE_LABEL
    )


def create_client() -> Any | None:
    """Создать клиент Anthropic, если пакет установлен.

    Ключ и адрес API SDK берёт из окружения (ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL).

    Returns:
        Клиент с таймаутом и числом повторов из конфига или None, если
        пакет anthropic не установлен.
    """
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
    )


def request_answer(note: str, client: Any) -> str | None:
    """Отправить наблюдение модели и вернуть текст ответа.

    Используются structured outputs (JSON-схема) и серверный fallback
    модели при отказе по политике безопасности.

    Args:
        note: Текст наблюдения.
        client: Клиент Anthropic или совместимая заглушка.

    Returns:
        Текст ответа или None, если модель не завершила ответ штатно.
    """
    response = client.beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=system_prompt(),
        messages=[
            {
                "role": "user",
                "content": config.LLM_USER_TEMPLATE.format(note=note),
            }
        ],
        output_config={
            "effort": config.LLM_EFFORT,
            "format": {"type": "json_schema", "schema": response_schema()},
        },
        betas=[config.LLM_FALLBACK_BETA],
        fallbacks=config.LLM_FALLBACKS,
    )
    if response.stop_reason != config.LLM_EXPECTED_STOP_REASON:
        return None
    parts = [block.text for block in response.content if block.type == "text"]
    return "".join(parts) or None


def parse_answer(raw: str, note: str) -> LLMAnswer | None:
    """Проверить формат ответа модели.

    Ответ отклоняется, если это не JSON-объект ровно с полями
    config.LLM_ANSWER_KEYS, типы полей неверны, зона не из списка,
    уверенность вне [0, 1], объяснение пустое, слишком длинное, не
    называет зону или содержит числа, которых нет в тексте наблюдения.

    Args:
        raw: Текст ответа модели.
        note: Исходное наблюдение.

    Returns:
        Проверенный ответ или None.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != set(config.LLM_ANSWER_KEYS):
        return None
    related, zone, confidence, explanation = (
        data[key] for key in config.LLM_ANSWER_KEYS
    )
    if not isinstance(related, bool) or not isinstance(zone, str):
        return None
    if not isinstance(explanation, str):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return None
    if not 0 <= confidence <= 1:
        return None
    text = explanation.strip()
    if not text or len(text) > config.LLM_MAX_EXPLANATION_CHARS:
        return None
    if related and (zone not in config.ZONES or f"'{zone}'" not in text):
        return None
    if not related and zone != config.LLM_NO_ZONE_LABEL:
        return None
    if not extract_numbers(text) <= extract_numbers(note):
        return None
    return LLMAnswer(
        is_zone=related,
        zone=zone if related else None,
        confidence=round(float(confidence), config.CONFIDENCE_DIGITS),
        explanation=text,
    )


def ask_llm(note: str, client: Any | None = None) -> LLMAnswer | None:
    """Спросить модель о зоне наблюдения, не выбрасывая исключений.

    Любая ошибка SDK или сети (нет ключа, неверный ключ, нет соединения,
    таймаут) пишется в лог уровня DEBUG и превращается в None: пайплайн
    должен работать и без LLM.

    Args:
        note: Текст наблюдения.
        client: Готовый клиент; если None, создаётся клиент Anthropic.

    Returns:
        Проверенный ответ модели или None.
    """
    try:
        active = client if client is not None else create_client()
        if active is None:
            return None
        raw = request_answer(note, active)
    except Exception as error:
        logger.debug("LLM недоступна, используются правила: %r", error)
        return None
    if raw is None:
        return None
    return parse_answer(raw, note)
