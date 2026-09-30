"""Общие фикстуры тестов: датасет из PDF и ручная разметка."""

import json
from pathlib import Path
from typing import Any

import pytest

from tutor_assistant import config


def _load(path: Path) -> Any:
    """Прочитать JSON-файл проекта.

    Args:
        path: Путь к файлу.

    Returns:
        Разобранное содержимое.
    """
    with path.open(encoding=config.FILE_ENCODING) as file:
        return json.load(file)


@pytest.fixture(scope="session")
def dataset() -> dict[str, Any]:
    """Вернуть содержимое dataset.json."""
    return _load(config.DATASET_PATH)


@pytest.fixture(scope="session")
def expected() -> dict[str, Any]:
    """Вернуть содержимое expected.json."""
    return _load(config.EXPECTED_PATH)


@pytest.fixture(scope="session")
def records(dataset: dict[str, Any]) -> dict[str, str]:
    """Вернуть тексты записей R1–R6 по идентификатору."""
    return {record["id"]: record["text"] for record in dataset["records"]}


@pytest.fixture(scope="session")
def gold_records(expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Вернуть разметку записей R1–R6 по идентификатору."""
    return {record["id"]: record for record in expected["records"]}
