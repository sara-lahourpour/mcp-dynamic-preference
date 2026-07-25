from __future__ import annotations

import argparse
import asyncio

from attacks.registry import get_condition_names
from host.prompts import get_policy_names
from host.runner import run_experiment


SERVER_IDS = [
    "server_1",
    "server_2",
    "server_3",
    "server_4",
    "server_5",
    "attacker",
]

ADVERSARIAL_CONDITION = "adversarial_memory"


def get_all_condition_names() -> list[str]:
    """
    Include the dynamic adversarial condition without requiring it
    to use the static attack-condition interface.
    """
    return sorted(
        set(
            get_condition_names()
            + [ADVERSARIAL_CONDITION]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=get_all_condition_names(),
        default="benign",
    )

    parser.add_argument(
        "--host-policy",
        choices=get_policy_names(),
        default="standard",
    )

    parser.add_argument(
        "--backend",
        choices=[
            "ollama",
            "openrouter",
        ],
        default="ollama",
    )

    parser.add_argument(
        "--attacker-server-id",
        choices=SERVER_IDS,
        default="attacker",
    )

    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Victim/client model used to select a tool.",
    )

    parser.add_argument(
        "--attacker-model",
        default="openai/gpt-4o-mini",
        help=(
            "OpenRouter model used as the adversarial "
            "metadata controller."
        ),
    )

    parser.add_argument(
        "--attacker-query-access",
        choices=[
            "none",
            "full",
        ],
        default="none",
        help=(
            "Whether the adversarial controller can see "
            "the current user query."
        ),
    )

    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help=(
            "Delete memory from a previous run before starting."
        ),
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=6,
        help=(
            "Repetitions for ordinary one-phase experiments. "
            "Ignored during --two-phase-memory-eval."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Victim/client model temperature.",
    )

    parser.add_argument(
        "--attacker-temperature",
        type=float,
        default=0.5,
        help=(
            "Adversarial controller temperature. Defaults to 0.5."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help=(
            "Maximum trials for ordinary one-phase experiments. "
            "Ignored during --two-phase-memory-eval."
        ),
    )

    parser.add_argument(
        "--attacker-custom-only",
        action="store_true",
        help=(
            "Restrict the adversarial controller to generating "
            "custom descriptions without fixed GAPMA strategies."
        ),
    )

    parser.add_argument(
        "--two-phase-memory-eval",
        action="store_true",
        help=(
            "Run memory training using 20 queries, then evaluate "
            "on 40 unseen queries with memory frozen."
        ),
    )

    parser.add_argument(
        "--stateless-heldout-eval",
        action="store_true",
        help=(
            "Evaluate the adversarial controller on the 40 held-out "
            "queries using empty, frozen memory and no training phase."
        ),
    )

    parser.add_argument(
        "--heldout-eval",
        action="store_true",
        help=(
            "Evaluate any condition on the same 40 held-out queries. "
            "Static baselines keep the configured attacker fixed."
        ),
    )

    parser.add_argument(
        "--truthfulness-policy",
        choices=[
            "truthful",
            "deceptive",
        ],
        default="truthful",
        help=(
            "Whether adversarial metadata must remain faithful to "
            "the implemented tool or may use unsupported claims."
        ),
    )

    parser.add_argument(
        "--resume-memory-heldout",
        action="store_true",
        help=(
            "Resume only the frozen-memory held-out phase after a "
            "completed two-phase training run. Does not reset memory."
        ),
    )

    parser.add_argument(
        "--training-repetitions",
        type=int,
        default=10,
        help=(
            "Number of repetitions for each training query in "
            "two-phase mode."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional unique result run ID. Existing run "
            "directories are never overwritten."
        ),
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    await run_experiment(
        condition_name=args.condition,
        attacker_server_id=args.attacker_server_id,
        host_policy=args.host_policy,
        backend_name=args.backend,
        model=args.model,
        repetitions=args.repetitions,
        temperature=args.temperature,
        seed=args.seed,
        max_trials=args.max_trials,
        attacker_model=args.attacker_model,
        attacker_temperature=args.attacker_temperature,
        reset_memory=args.reset_memory,
        attacker_query_access=args.attacker_query_access,
        attacker_custom_only=args.attacker_custom_only,
        truthfulness_policy=args.truthfulness_policy,
        two_phase_memory_eval=args.two_phase_memory_eval,
        training_repetitions=args.training_repetitions,
        stateless_heldout_eval=(
            args.stateless_heldout_eval
        ),
        heldout_eval=args.heldout_eval,
        resume_memory_heldout=args.resume_memory_heldout,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    asyncio.run(main())
