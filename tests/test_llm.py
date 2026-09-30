"""Тесты LLM-пути check_zone: валидный ответ принимается, любая проблема —
тихий переход на правила.

Сеть не нужна: вместо Anthropic API поднимается локальный HTTP-сервер,
а для проверки формата используются заглушки клиента.
"""

import json
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from tutor_assistant import config
from tutor_assistant.zones import check_zone_detailed, check_zone_rules

NOTE = "отказ от еды третий день подряд"
VALID_ANSWER: dict[str, Any] = {
    "related": True,
    "zone": "питание",
    "confidence": 0.9,
    "explanation": "Отказ от еды третий день подряд относится к зоне "
                   "'питание'.",
}
AUTH_ERROR_BODY: dict[str, Any] = {
    "type": "error",
    "error": {"type": "authentication_error", "message": "invalid x-api-key"},
}
INVALID_KEY = "sk-ant-invalid-key-for-tests"
LOCALHOST = "127.0.0.1"
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_PROFILE",
)
HOME_ENV = ("HOME", "USERPROFILE", "APPDATA", "XDG_CONFIG_HOME")

Received = list[dict[str, str]]
ServerFactory = Callable[[int, dict[str, Any]], tuple[str, Received]]


class FakeMessages:
    """Заглушка client.beta.messages с заранее заданным ответом."""

    def __init__(
        self, response: Any = None, error: Exception | None = None
    ) -> None:
        """Запомнить ответ или исключение для create.

        Args:
            response: Объект ответа.
            error: Исключение, которое нужно выбросить.
        """
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Записать параметры вызова и вернуть заготовленный ответ.

        Args:
            **kwargs: Параметры запроса.

        Returns:
            Заготовленный ответ.
        """
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def fake_client(
    text: str, stop_reason: str = config.LLM_EXPECTED_STOP_REASON
) -> SimpleNamespace:
    """Собрать заглушку клиента, возвращающую текстовый ответ.

    Args:
        text: Текст ответа модели.
        stop_reason: Причина остановки.

    Returns:
        Объект с атрибутом beta.messages.
    """
    response = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def answer(**changes: Any) -> str:
    """Собрать JSON-ответ модели с изменёнными полями.

    Args:
        **changes: Поля, которые нужно заменить.

    Returns:
        JSON-строка.
    """
    return json.dumps({**VALID_ANSWER, **changes}, ensure_ascii=False)


def message_body(text: str) -> dict[str, Any]:
    """Собрать тело успешного ответа Messages API.

    Args:
        text: Текст ответа модели.

    Returns:
        JSON-объект сообщения.
    """
    return {
        "id": "msg_local_test",
        "type": "message",
        "role": "assistant",
        "model": config.LLM_MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": config.LLM_EXPECTED_STOP_REASON,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убрать учётные данные Anthropic и профили из окружения теста."""
    for name in CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    missing_home = str(config.PROJECT_ROOT / "missing-home-for-tests")
    for name in HOME_ENV:
        monkeypatch.setenv(name, missing_home)


@pytest.fixture
def api_server() -> Iterator[ServerFactory]:
    """Запускать локальные серверы, имитирующие Anthropic API."""
    servers: list[ThreadingHTTPServer] = []

    def start(
        status: int, body: dict[str, Any]
    ) -> tuple[str, Received]:
        """Запустить сервер с фиксированным ответом.

        Args:
            status: HTTP-статус ответа.
            body: JSON-тело ответа.

        Returns:
            Базовый URL сервера и список заголовков полученных запросов.
        """
        received: Received = []
        payload = json.dumps(body).encode(config.FILE_ENCODING)

        class Handler(BaseHTTPRequestHandler):
            """Обработчик, отвечающий заготовленным JSON."""

            def do_POST(self) -> None:
                """Прочитать запрос и вернуть заготовленный ответ."""
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                headers = self.headers.items()
                received.append(
                    {key.lower(): value for key, value in headers}
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                """Не писать журнал запросов в консоль."""

        server = ThreadingHTTPServer((LOCALHOST, 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://{LOCALHOST}:{server.server_address[1]}", received

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_valid_answer_is_used() -> None:
    """Корректный ответ модели возвращается с источником LLM."""
    client = fake_client(answer())
    result = check_zone_detailed(NOTE, use_llm=True, client=client)
    assert result.source == config.SOURCE_LLM
    assert result.as_tuple() == (True, 0.9, VALID_ANSWER["explanation"])


def test_request_uses_config() -> None:
    """В запрос уходят модель из конфига, JSON-схема и текст наблюдения."""
    client = fake_client(answer())
    check_zone_detailed(NOTE, use_llm=True, client=client)
    call = client.beta.messages.calls[0]
    assert call["model"] == config.LLM_MODEL
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert NOTE in call["messages"][0]["content"]


def test_llm_is_off_by_default() -> None:
    """Без явного включения LLM не вызывается."""
    client = fake_client(answer())
    result = check_zone_detailed(NOTE, client=client)
    assert result.source == config.SOURCE_RULES
    assert client.beta.messages.calls == []


@pytest.mark.parametrize(
    "raw",
    [
        "не JSON",
        json.dumps([VALID_ANSWER]),
        json.dumps({**VALID_ANSWER, "extra": 1}),
        answer(zone="игра"),
        answer(confidence=1.5),
        answer(confidence=True),
        answer(related="да"),
        answer(explanation=""),
        answer(explanation="Отказ от еды относится к питанию."),
        answer(explanation="Отказ от еды 5 дней: зона 'питание'."),
        answer(related=False),
    ],
    ids=["not_json", "list", "extra_key", "unknown_zone", "confidence_gt_1",
         "confidence_bool", "related_str", "empty_explanation",
         "zone_not_named", "invented_number", "false_with_zone"],
)
def test_malformed_answer_falls_back(raw: str) -> None:
    """Ответ не того формата молча заменяется ответом правил."""
    result = check_zone_detailed(NOTE, use_llm=True, client=fake_client(raw))
    assert result == check_zone_rules(NOTE)


def test_refusal_falls_back() -> None:
    """Отказ модели (stop_reason=refusal) ведёт к правилам."""
    client = fake_client(answer(), stop_reason="refusal")
    result = check_zone_detailed(NOTE, use_llm=True, client=client)
    assert result == check_zone_rules(NOTE)


def test_client_error_falls_back_silently(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Исключение клиента не выходит наружу и ничего не печатает."""
    messages = FakeMessages(error=RuntimeError("сеть недоступна"))
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    result = check_zone_detailed(NOTE, use_llm=True, client=client)
    assert result == check_zone_rules(NOTE)
    assert capsys.readouterr() == ("", "")


def test_missing_anthropic_package_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без установленного пакета anthropic работают правила."""
    monkeypatch.setitem(sys.modules, "anthropic", None)
    result = check_zone_detailed(NOTE, use_llm=True)
    assert result == check_zone_rules(NOTE)


def test_no_key_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: None,
    api_server: ServerFactory,
) -> None:
    """Без ключа LLM-режим молча уступает правилам."""
    pytest.importorskip("anthropic")
    base_url, _ = api_server(HTTP_UNAUTHORIZED, AUTH_ERROR_BODY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    result = check_zone_detailed(NOTE, use_llm=True)
    assert result == check_zone_rules(NOTE)


def test_invalid_key_falls_back_silently(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    isolated_env: None,
    api_server: ServerFactory,
) -> None:
    """Неверный ключ: настоящий SDK получает 401, ответ берётся из правил."""
    pytest.importorskip("anthropic")
    base_url, received = api_server(HTTP_UNAUTHORIZED, AUTH_ERROR_BODY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", INVALID_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    result = check_zone_detailed(NOTE, use_llm=True)
    assert result == check_zone_rules(NOTE)
    assert [request["x-api-key"] for request in received] == [INVALID_KEY]
    assert capsys.readouterr() == ("", "")


def test_unreachable_api_falls_back(
    monkeypatch: pytest.MonkeyPatch, isolated_env: None
) -> None:
    """Недоступная сеть: ошибка соединения, ответ берётся из правил."""
    pytest.importorskip("anthropic")
    with socket.socket() as probe:
        probe.bind((LOCALHOST, 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("ANTHROPIC_API_KEY", INVALID_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://{LOCALHOST}:{port}")
    result = check_zone_detailed(NOTE, use_llm=True)
    assert result == check_zone_rules(NOTE)


def test_valid_answer_through_real_sdk(
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: None,
    api_server: ServerFactory,
) -> None:
    """Настоящий SDK принимает параметры запроса и разбирает ответ."""
    pytest.importorskip("anthropic")
    base_url, received = api_server(HTTP_OK, message_body(answer()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", INVALID_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    result = check_zone_detailed(NOTE, use_llm=True)
    assert result.source == config.SOURCE_LLM
    assert result.zone == "питание"
    assert len(received) == 1
