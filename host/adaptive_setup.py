from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from attacks.adversarial_assets import (
    load_adversarial_descriptions,
)
from attacks.adversarial_controller import (
    AdversarialController,
)
from attacks.adversarial_memory import AttackerMemory
from host.experiment_config import ENV_PATH
from host.memory_paths import build_memory_path


load_dotenv(ENV_PATH)


@dataclass
class RuntimeDescription:
    description: str
    lesson: str = ""


@dataclass
class AdaptiveRuntime:
    controller: AdversarialController
    memory: AttackerMemory
    memory_path: Path
    held_out_description: RuntimeDescription | None = None
    evaluation_memory: dict[str, Any] | None = None
    full_evaluation_evidence: dict[str, Any] | None = None
    training_description: str = ""
    training_lesson: str = ""
    block_start_description: str = ""
    block_start_lesson: str = ""
    block_outcomes: list[dict[str, Any]] = field(
        default_factory=list
    )


def create_openrouter_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY was not found in .env."
        )

    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": (
                "http://localhost/"
                "mcp-preference-research"
            ),
            "X-Title": "MCP Preference Research",
        },
    )


def create_adaptive_runtime(
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
    original_description: str,
    reset_memory: bool,
) -> AdaptiveRuntime:
    memory_path = build_memory_path(
        backend_name=backend_name,
        victim_model=victim_model,
        host_policy=host_policy,
        attacker_server_id=attacker_server_id,
        attacker_model=attacker_model,
        attacker_query_access=attacker_query_access,
        attacker_custom_only=attacker_custom_only,
        truthfulness_policy=truthfulness_policy,
        attacker_temperature=attacker_temperature,
    )

    if reset_memory and memory_path.exists():
        memory_path.unlink()

    memory = AttackerMemory.load(memory_path)

    controller = AdversarialController(
        client=create_openrouter_client(),
        model=attacker_model,
        original_description=original_description,
        fixed_descriptions=load_adversarial_descriptions(),
        max_recent_trials=10,
        custom_only=attacker_custom_only,
        truthfulness_policy=truthfulness_policy,
        temperature=attacker_temperature,
    )

    return AdaptiveRuntime(
        controller=controller,
        memory=memory,
        memory_path=memory_path,
    )
