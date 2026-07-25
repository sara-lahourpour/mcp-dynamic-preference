from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GAPMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "gapma_descriptions.json"
)

GA_PATH = (
    PROJECT_ROOT
    / "data"
    / "gapma_ga_descriptions.json"
)


def load_adversarial_descriptions() -> dict[str, str]:
    descriptions: dict[str, str] = {}

    with GAPMA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        gapma_data = json.load(file)

    for strategy, description in gapma_data.get(
        "descriptions",
        {},
    ).items():
        if isinstance(description, str):
            descriptions[strategy] = description.strip()

    with GA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        ga_data = json.load(file)

    optimized = ga_data.get(
        "optimized_strategies",
        {},
    ).get(
        "exaggerated",
        {},
    ).get(
        "optimized_description"
    )

    if isinstance(optimized, str) and optimized.strip():
        descriptions["ga_optimized"] = optimized.strip()

    return descriptions