from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVER_SCRIPT = (
    PROJECT_ROOT
    / "servers"
    / "equivalent_tool_server.py"
)

COMPETITORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "competitors.json"
)

QUERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "queries.json"
)

TRAIN_QUERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "train_queries.json"
)

TEST_QUERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "test_queries.json"
)

ENV_PATH = PROJECT_ROOT / ".env"

ADVERSARIAL_CONDITION = "adversarial_memory"


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_competitors() -> list[dict[str, Any]]:
    data = load_json(COMPETITORS_PATH)

    if not isinstance(data, list):
        raise ValueError(
            "data/competitors.json must contain a list."
        )

    return data


def _validate_queries(
    *,
    data: Any,
    expected_count: int,
    file_name: str,
) -> list[str]:
    if not isinstance(data, list):
        raise ValueError(
            f"{file_name} must contain a JSON list."
        )

    queries = [
        " ".join(
            str(query).strip().split()
        )
        for query in data
    ]

    if any(not query for query in queries):
        raise ValueError(
            f"{file_name} contains an empty query."
        )

    normalized = [
        query.casefold()
        for query in queries
    ]

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"{file_name} contains duplicate queries."
        )

    if len(queries) != expected_count:
        raise ValueError(
            f"{file_name} must contain exactly "
            f"{expected_count} queries, "
            f"but found {len(queries)}."
        )

    return queries


def load_queries() -> list[str]:
    return _validate_queries(
        data=load_json(QUERIES_PATH),
        expected_count=60,
        file_name="data/queries.json",
    )


def load_train_queries() -> list[str]:
    return _validate_queries(
        data=load_json(TRAIN_QUERIES_PATH),
        expected_count=20,
        file_name="data/train_queries.json",
    )


def load_test_queries() -> list[str]:
    return _validate_queries(
        data=load_json(TEST_QUERIES_PATH),
        expected_count=40,
        file_name="data/test_queries.json",
    )


def validate_train_test_split(
    train_queries: list[str],
    test_queries: list[str],
) -> None:
    train_normalized = {
        query.casefold()
        for query in train_queries
    }

    test_normalized = {
        query.casefold()
        for query in test_queries
    }

    overlap = (
        train_normalized
        & test_normalized
    )

    if overlap:
        raise ValueError(
            "Training and held-out query sets overlap: "
            + ", ".join(sorted(overlap))
        )