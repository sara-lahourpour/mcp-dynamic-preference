from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"


CONDITION_FOLDERS = {
    "benign": (
        "stage_01_benign",
        None,
    ),
    "dpma_best_name": (
        "stage_02_dpma",
        "best_name",
    ),
    "dpma_best_description": (
        "stage_02_dpma",
        "best_description",
    ),

    # These will be used later.
    "gapma_authoritative": (
        "stage_03_gapma",
        "authoritative",
    ),
    "gapma_emotional": (
        "stage_03_gapma",
        "emotional",
    ),
    "gapma_exaggerated": (
        "stage_03_gapma",
        "exaggerated",
    ),
    "gapma_subliminal": (
        "stage_03_gapma",
        "subliminal",
    ),
    "gapma_ga_optimized": (
        "stage_03_gapma",
        "ga_optimized",
    ),
    "adversarial_memory": (
        "stage_04_adversarial_memory",
        None,
    ),
    "dynamic_metadata": (
        "stage_04_dynamic_metadata",
        None,
    ),
    "adaptive_attacker": (
        "stage_05_adaptive_attacker",
        None,
    ),
}


@dataclass(frozen=True)
class ResultPaths:
    directory: Path
    trials: Path
    summary: Path


def safe_path_component(value: str) -> str:
    """
    Convert values such as qwen/qwen3-32b into a safe folder name.
    """
    safe = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value.strip(),
    ).strip("._")
    if not safe:
        raise ValueError("Path component cannot be empty.")
    return safe


def create_run_id(
    *,
    mode: str,
    truthfulness_policy: str | None,
    seed: int,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    policy = (
        safe_path_component(truthfulness_policy)
        if truthfulness_policy
        else "na"
    )
    suffix = uuid4().hex[:8]
    return (
        f"{timestamp}_{safe_path_component(mode)}_"
        f"{policy}_s{seed}_{suffix}"
    )


def build_unique_run_paths(
    *,
    condition: str,
    run_id: str,
) -> ResultPaths:
    try:
        stage_folder, _ = CONDITION_FOLDERS[condition]
    except KeyError as error:
        raise ValueError(
            f"No result folder is configured for condition: "
            f"{condition}"
        ) from error

    directory = (
        RESULTS_ROOT
        / stage_folder
        / "runs"
        / safe_path_component(run_id)
    )
    directory.mkdir(parents=True, exist_ok=False)

    return ResultPaths(
        directory=directory,
        trials=directory / "trials.jsonl",
        summary=directory / "summary.json",
    )


def build_result_paths(
    *,
    condition: str,
    backend: str,
    model: str,
    host_policy: str,
    attacker_server_id: str,
) -> ResultPaths:
    try:
        stage_folder, condition_folder = CONDITION_FOLDERS[
            condition
        ]
    except KeyError as error:
        raise ValueError(
            f"No result folder is configured for condition: "
            f"{condition}"
        ) from error

    directory = (
        RESULTS_ROOT
        / stage_folder
    )

    if condition_folder is not None:
        directory = directory / condition_folder

    directory = (
        directory
        / safe_path_component(backend)
        / safe_path_component(model)
        / safe_path_component(host_policy)
        / safe_path_component(attacker_server_id)
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return ResultPaths(
        directory=directory,
        trials=directory / "trials.jsonl",
        summary=directory / "summary.json",
    )
