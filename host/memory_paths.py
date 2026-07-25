from __future__ import annotations

from pathlib import Path

from host.experiment_config import PROJECT_ROOT


def safe_path_component(value: str) -> str:
    result = value.strip()

    for character in [
        ":",
        "/",
        "\\",
        " ",
        ".",
    ]:
        result = result.replace(character, "_")

    return result


def build_memory_path(
    *,
    backend_name: str,
    victim_model: str,
    host_policy: str,
    attacker_server_id: str,
    attacker_model: str,
    attacker_temperature: float,
    attacker_query_access: str,
    attacker_custom_only: bool,
    truthfulness_policy: str,
) -> Path:
    adaptive_variant = (
        f"{attacker_query_access}_custom_only"
        if attacker_custom_only
        else f"{attacker_query_access}_mixed"
    )

    return (
        PROJECT_ROOT
        / "memory"
        / safe_path_component(backend_name)
        / safe_path_component(victim_model)
        / safe_path_component(host_policy)
        / safe_path_component(attacker_server_id)
        / safe_path_component(attacker_model)
        / safe_path_component(
            f"temperature_{attacker_temperature:g}"
        )
        / safe_path_component(adaptive_variant)
        / safe_path_component(truthfulness_policy)
        / "memory.json"
    )
