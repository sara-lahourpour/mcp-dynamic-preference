from __future__ import annotations

import json
import random
import traceback
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from attacks.adversarial_controller import ROLLING_MAX_OUTPUT_TOKENS
from attacks.registry import get_condition
from backends.registry import create_backend
from host.adaptive_setup import (
    AdaptiveRuntime,
    RuntimeDescription,
    create_adaptive_runtime,
)
from host.experiment_config import (
    ADVERSARIAL_CONDITION,
    load_competitors,
    load_queries,
    load_test_queries,
    load_train_queries,
    validate_train_test_split,
)
from host.evaluation_memory import (
    build_compact_evaluation_memory,
    build_full_evaluation_evidence,
)
from host.prompts import get_system_prompt
from host.result_paths import (
    ResultPaths,
    build_unique_run_paths,
    create_run_id,
)
from host.result_writer import (
    JsonlWriter,
    write_json,
)
from host.tool_setup import (
    connect_experiment_tools,
)
from host.trial_executor import execute_trial


ROLLING_QUERY_BLOCK_SIZE = 10
REQUIRED_ROLLING_TRAINING_TRIALS = 200


@dataclass
class PhaseCounters:
    completed: int = 0
    valid: int = 0
    invalid: int = 0
    attacker_selections: int = 0

    @property
    def asr(self) -> float:
        if self.valid == 0:
            return 0.0

        return (
            self.attacker_selections
            / self.valid
        )


def build_balanced_attacker_schedule(
    *,
    server_ids: list[str],
    total_trials: int,
    seed: int,
) -> list[str]:
    if not server_ids:
        raise ValueError(
            "Cannot rotate attackers without server IDs."
        )

    if total_trials < 1:
        return []

    rng = random.Random(seed)
    schedule: list[str] = []

    while len(schedule) < total_trials:
        block = list(server_ids)
        rng.shuffle(block)
        schedule.extend(block)

    return schedule[:total_trials]


def get_unique_server_ids(
    tools: list[Any],
) -> list[str]:
    return list(
        dict.fromkeys(
            tool.server_id
            for tool in tools
        )
    )


def build_training_sequence(
    *,
    queries: list[str],
    repetitions: int,
    seed: int,
) -> list[tuple[int, str]]:
    """
    Return (repetition, query) pairs.

    Each repetition contains every training query exactly once,
    but query order is independently shuffled.
    """
    rng = random.Random(seed)
    sequence: list[tuple[int, str]] = []

    for repetition in range(
        1,
        repetitions + 1,
    ):
        block = list(queries)
        rng.shuffle(block)

        sequence.extend(
            (
                repetition,
                query,
            )
            for query in block
        )

    return sequence


def build_rolling_training_sequence(
    *,
    queries: list[str],
    repetitions: int,
    seed: int,
) -> list[tuple[int, str]]:
    """
    Shuffle query order once, then keep each query's repetitions
    contiguous so immediate feedback forms one query block.
    """
    rng = random.Random(seed)
    ordered_queries = list(queries)
    rng.shuffle(ordered_queries)

    return [
        (repetition, query)
        for query in ordered_queries
        for repetition in range(1, repetitions + 1)
    ]


def build_evaluation_sequence(
    *,
    queries: list[str],
    seed: int,
) -> list[tuple[int, str]]:
    """
    Held-out queries are each evaluated exactly once.
    """
    rng = random.Random(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)

    return [
        (1, query)
        for query in shuffled
    ]


def update_phase_counters(
    *,
    counters: PhaseCounters,
    outcome: Any,
) -> None:
    counters.completed += 1

    if outcome.valid:
        counters.valid += 1

        if outcome.attacker_selected:
            counters.attacker_selections += 1
    else:
        counters.invalid += 1


async def create_held_out_description(
    *,
    adaptive_runtime: AdaptiveRuntime,
) -> None:
    """
    Make one attacker-LLM call using frozen training memory.

    This synthesized description is runtime-only: held-out evaluation
    neither adds it to persistent memory nor writes outcomes back.
    """
    full_evidence = build_full_evaluation_evidence(
        adaptive_runtime.memory
    )
    evaluation_memory = build_compact_evaluation_memory(
        full_evidence
    )
    adaptive_runtime.full_evaluation_evidence = full_evidence

    try:
        lesson = (
            await adaptive_runtime.controller
            .generate_evaluation_memory_lesson(
                examples=evaluation_memory,
            )
        )
        evaluation_memory["lesson"] = lesson
        adaptive_runtime.evaluation_memory = evaluation_memory
        decision = (
            await adaptive_runtime.controller
            .generate_evaluation_description(
                memory=evaluation_memory,
            )
        )
    except Exception as error:
        print()
        print(
            "FINAL MEMORY SYNTHESIS ERROR: "
            f"{type(error).__name__}: {error}"
        )
        print(
            "Training memory is preserved. Resume again without "
            "--reset-memory."
        )
        raise

    adaptive_runtime.held_out_description = RuntimeDescription(
        description=decision.description,
    )


async def create_stateless_held_out_description(
    *,
    adaptive_runtime: AdaptiveRuntime,
) -> None:
    """
    Generate one description without training memory and freeze it.
    """
    adaptive_runtime.evaluation_memory = None
    adaptive_runtime.full_evaluation_evidence = None
    decision = (
        await adaptive_runtime.controller
        .generate_evaluation_description(memory=None)
    )
    adaptive_runtime.held_out_description = RuntimeDescription(
        description=decision.description,
    )


def record_final_adversarial_prompt(
    *,
    adaptive_runtime: AdaptiveRuntime,
    result_paths: ResultPaths,
    evaluation_mode: str,
    truthfulness_policy: str,
) -> None:
    request = adaptive_runtime.controller.last_rolling_request
    if request is None:
        raise RuntimeError(
            "The final adversarial request was not captured."
        )

    held_out = adaptive_runtime.held_out_description
    if held_out is None:
        raise RuntimeError(
            "The final evaluation description was not generated."
        )

    memory_supplied = adaptive_runtime.evaluation_memory
    memory_record = {
        "evaluation_mode": evaluation_mode,
        "truthfulness_policy": truthfulness_policy,
        "memory_supplied_to_final_adversarial_llm": (
            memory_supplied
        ),
    }
    memory_output_path = (
        result_paths.directory
        / "eval_memory.json"
    )
    write_json(
        path=memory_output_path,
        data=memory_record,
    )

    full_evidence_path = (
        result_paths.directory
        / "full_evidence.json"
    )
    full_evidence_record = {
        "evaluation_mode": evaluation_mode,
        "truthfulness_policy": truthfulness_policy,
        "full_training_evidence": (
            adaptive_runtime.full_evaluation_evidence
        ),
    }
    write_json(
        path=full_evidence_path,
        data=full_evidence_record,
    )

    record = {
        "evaluation_mode": evaluation_mode,
        "truthfulness_policy": truthfulness_policy,
        "memory_supplied_to_final_adversarial_llm": (
            memory_supplied
        ),
        "memory_lesson_synthesis_request": (
            adaptive_runtime
            .controller
            .last_memory_lesson_request
        ),
        "request": request,
        "truthful_fallback_used": (
            adaptive_runtime
            .controller
            .last_rolling_used_fallback
        ),
        "generated_description": held_out.description,
    }
    output_path = (
        result_paths.directory
        / "final_prompt.json"
    )
    write_json(
        path=output_path,
        data=record,
    )

    print()
    print("=" * 72)
    print("MEMORY SUPPLIED BEFORE EVALUATION")
    print("=" * 72)
    print(
        json.dumps(
            memory_record,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved to: {memory_output_path}")
    print(
        "Full audit evidence saved to: "
        f"{full_evidence_path}"
    )
    print("=" * 72)
    print()
    print("=" * 72)
    print("FINAL ADVERSARIAL PROMPT BEFORE EVALUATION")
    print("=" * 72)
    console_request = dict(request)
    console_messages = [
        dict(message)
        for message in request["messages"]
    ]
    user_payload = json.loads(
        console_messages[1]["content"]
    )
    if "memory" in user_payload:
        user_payload["memory"] = (
            "<same compact memory printed above>"
        )
    console_messages[1]["content"] = json.dumps(
        user_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    console_request["messages"] = console_messages
    console_record = {
        "evaluation_mode": evaluation_mode,
        "truthfulness_policy": truthfulness_policy,
        "request": console_request,
        "truthful_fallback_used": record[
            "truthful_fallback_used"
        ],
        "generated_description": held_out.description,
        "note": (
            "The exact unabridged request is stored in the file below."
        ),
    }
    print(
        json.dumps(
            console_record,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved to: {output_path}")
    print("=" * 72)


async def create_and_record_memory_evaluation(
    *,
    adaptive_runtime: AdaptiveRuntime,
    result_paths: ResultPaths,
    truthfulness_policy: str,
) -> None:
    try:
        await create_held_out_description(
            adaptive_runtime=adaptive_runtime
        )
        record_final_adversarial_prompt(
            adaptive_runtime=adaptive_runtime,
            result_paths=result_paths,
            evaluation_mode="memory_trained",
            truthfulness_policy=truthfulness_policy,
        )
    except BaseException as error:
        error_record = {
            "stage": (
                "final_memory_synthesis_or_audit_before_evaluation"
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        error_path = (
            result_paths.directory
            / "pre_eval_error.json"
        )
        write_json(
            path=error_path,
            data=error_record,
        )
        print()
        print("=" * 72, flush=True)
        print(
            "PRE-EVALUATION ERROR: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        print(
            f"Full root traceback saved to: {error_path}",
            flush=True,
        )
        print("=" * 72, flush=True)
        raise


async def prepare_rolling_training_trial(
    *,
    adaptive_runtime: AdaptiveRuntime,
    phase_trial_index: int,
) -> None:
    is_block_start = (
        (phase_trial_index - 1)
        % ROLLING_QUERY_BLOCK_SIZE
        == 0
    )

    if not is_block_start:
        return

    if not adaptive_runtime.memory.active_description:
        decision = (
            await adaptive_runtime.controller
            .generate_initial_rolling_description()
        )
        adaptive_runtime.memory.active_description = (
            decision.description
        )
        adaptive_runtime.memory.active_lesson = ""

    adaptive_runtime.training_description = (
        adaptive_runtime.memory.active_description
    )
    adaptive_runtime.training_lesson = (
        adaptive_runtime.memory.active_lesson
    )
    adaptive_runtime.block_start_description = (
        adaptive_runtime.training_description
    )
    adaptive_runtime.block_start_lesson = (
        adaptive_runtime.training_lesson
    )
    adaptive_runtime.block_outcomes = []


async def update_rolling_training_state(
    *,
    adaptive_runtime: AdaptiveRuntime,
    outcome: Any,
    phase_trial_index: int,
) -> None:
    is_block_end = (
        phase_trial_index
        % ROLLING_QUERY_BLOCK_SIZE
        == 0
    )

    if outcome.valid and outcome.description:
        adaptive_runtime.block_outcomes.append(
            {
                "description": outcome.description,
                "worked": outcome.attacker_selected,
            }
        )

    if is_block_end:
        if not adaptive_runtime.block_outcomes:
            raise RuntimeError(
                "Cannot summarize a query block with no valid trials."
            )
        decision = (
            await adaptive_runtime.controller.summarize_query_block(
                previous_description=(
                    adaptive_runtime.block_start_description
                ),
                previous_lesson=(
                    adaptive_runtime.block_start_lesson
                ),
                outcomes=adaptive_runtime.block_outcomes,
                prior_block_memory=[
                    {
                        "block": summary.query_block,
                        "description": summary.description,
                        "lesson": summary.lesson,
                        "selected": summary.successes,
                        "trials": summary.attempts,
                    }
                    for summary
                    in adaptive_runtime.memory.query_summaries
                ],
            )
        )
        block_number = (
            phase_trial_index
            // ROLLING_QUERY_BLOCK_SIZE
        )
        successes = sum(
            int(item["worked"])
            for item in adaptive_runtime.block_outcomes
        )
        adaptive_runtime.memory.add_query_summary(
            query_block=block_number,
            description=decision.description,
            lesson=decision.lesson,
            attempts=len(adaptive_runtime.block_outcomes),
            successes=successes,
        )
        adaptive_runtime.memory.save(
            adaptive_runtime.memory_path
        )
        return

    if (
        outcome.valid
        and outcome.description
        and not outcome.attacker_selected
    ):
        decision = (
            await adaptive_runtime.controller.revise_after_failure(
                description=adaptive_runtime.training_description,
                lesson=adaptive_runtime.training_lesson,
            )
        )
        adaptive_runtime.training_description = (
            decision.description
        )


async def run_experiment(
    *,
    condition_name: str,
    attacker_server_id: str,
    host_policy: str,
    backend_name: str,
    model: str,
    repetitions: int,
    temperature: float,
    seed: int,
    max_trials: int | None,
    attacker_model: str | None = None,
    attacker_temperature: float = 0.5,
    reset_memory: bool = False,
    attacker_query_access: str,
    attacker_custom_only: bool = False,
    truthfulness_policy: str = "truthful",
    two_phase_memory_eval: bool = False,
    training_repetitions: int = 10,
    stateless_heldout_eval: bool = False,
    heldout_eval: bool = False,
    resume_memory_heldout: bool = False,
    run_id: str | None = None,
) -> None:
    random.seed(seed)

    competitors = load_competitors()

    is_adversarial = (
        condition_name
        == ADVERSARIAL_CONDITION
    )

    selected_eval_modes = sum(
        [
            bool(two_phase_memory_eval),
            bool(stateless_heldout_eval),
            bool(heldout_eval),
            bool(resume_memory_heldout),
        ]
    )

    if selected_eval_modes > 1:
        raise ValueError(
            "--two-phase-memory-eval, --stateless-heldout-eval, "
            "--heldout-eval, and --resume-memory-heldout are "
            "mutually exclusive."
        )

    if (
        (
            two_phase_memory_eval
            or stateless_heldout_eval
            or resume_memory_heldout
        )
        and not is_adversarial
    ):
        raise ValueError(
            "Memory evaluation modes require "
            "--condition adversarial_memory."
        )

    if (
        stateless_heldout_eval
        and not reset_memory
    ):
        raise ValueError(
            "--stateless-heldout-eval requires --reset-memory "
            "so evaluation starts with empty memory."
        )

    if resume_memory_heldout and reset_memory:
        raise ValueError(
            "--resume-memory-heldout cannot be combined with "
            "--reset-memory."
        )

    if is_adversarial and not attacker_model:
        raise ValueError(
            "--attacker-model is required for "
            "adversarial_memory."
        )

    if (
        two_phase_memory_eval
        and training_repetitions < 1
    ):
        raise ValueError(
            "--training-repetitions must be at least 1."
        )

    if resume_memory_heldout:
        test_queries = load_test_queries()

        evaluation_sequence = (
            build_evaluation_sequence(
                queries=test_queries,
                seed=seed + 2_000,
            )
        )

        phases = [
            {
                "name": "held_out_evaluation",
                "sequence": evaluation_sequence,
                "update_memory": False,
            }
        ]

    elif stateless_heldout_eval:
        test_queries = load_test_queries()

        evaluation_sequence = (
            build_evaluation_sequence(
                queries=test_queries,
                seed=seed + 2_000,
            )
        )

        phases = [
            {
                "name": "stateless_held_out_evaluation",
                "sequence": evaluation_sequence,
                "update_memory": False,
            }
        ]

    elif heldout_eval:
        test_queries = load_test_queries()

        evaluation_sequence = (
            build_evaluation_sequence(
                queries=test_queries,
                seed=seed + 2_000,
            )
        )

        phases = [
            {
                "name": "held_out_evaluation",
                "sequence": evaluation_sequence,
                "update_memory": False,
            }
        ]

    elif two_phase_memory_eval:
        train_queries = load_train_queries()
        test_queries = load_test_queries()

        validate_train_test_split(
            train_queries,
            test_queries,
        )

        training_sequence = build_rolling_training_sequence(
            queries=train_queries,
            repetitions=training_repetitions,
            seed=seed + 1_000,
        )

        if (
            len(training_sequence)
            != REQUIRED_ROLLING_TRAINING_TRIALS
        ):
            raise ValueError(
                "Rolling memory design requires exactly "
                f"{REQUIRED_ROLLING_TRAINING_TRIALS} "
                "training trials. Current configuration "
                f"produces {len(training_sequence)}."
            )

        evaluation_sequence = (
            build_evaluation_sequence(
                queries=test_queries,
                seed=seed + 2_000,
            )
        )

        phases = [
            {
                "name": "training",
                "sequence": training_sequence,
                "update_memory": True,
            },
            {
                "name": "held_out_evaluation",
                "sequence": evaluation_sequence,
                "update_memory": False,
            },
        ]

    else:
        queries = load_queries()

        ordinary_sequence = (
            build_training_sequence(
                queries=queries,
                repetitions=repetitions,
                seed=seed + 1_000,
            )
        )

        if max_trials is not None:
            ordinary_sequence = (
                ordinary_sequence[:max_trials]
            )

        phases = [
            {
                "name": "ordinary",
                "sequence": ordinary_sequence,
                "update_memory": True,
            }
        ]

    backend = create_backend(backend_name)

    system_prompt = get_system_prompt(
        host_policy
    )

    rotate_attacker = is_adversarial

    static_condition = (
        None
        if is_adversarial
        else get_condition(condition_name)
    )

    if stateless_heldout_eval:
        run_mode = "stateless"
    elif two_phase_memory_eval:
        run_mode = "memory"
    elif resume_memory_heldout:
        run_mode = "memory_resume"
    elif heldout_eval:
        run_mode = "heldout"
    else:
        run_mode = "ordinary"

    resolved_run_id = run_id or create_run_id(
        mode=run_mode,
        truthfulness_policy=(
            truthfulness_policy if is_adversarial else None
        ),
        seed=seed,
    )
    result_paths = build_unique_run_paths(
        condition=condition_name,
        run_id=resolved_run_id,
    )

    started_at_utc = datetime.now(timezone.utc).isoformat()
    run_manifest = {
        "schema_version": 1,
        "run_id": resolved_run_id,
        "status": "running",
        "started_at_utc": started_at_utc,
        "condition": condition_name,
        "run_mode": run_mode,
        "backend": backend_name,
        "victim_model": model,
        "attacker_model": attacker_model,
        "host_policy": host_policy,
        "victim_temperature": temperature,
        "attacker_temperature": (
            attacker_temperature if is_adversarial else None
        ),
        "attacker_max_output_tokens": (
            ROLLING_MAX_OUTPUT_TOKENS if is_adversarial else None
        ),
        "seed": seed,
        "repetitions": repetitions,
        "training_repetitions": (
            training_repetitions
            if two_phase_memory_eval or resume_memory_heldout
            else None
        ),
        "truthfulness_policy": (
            truthfulness_policy if is_adversarial else None
        ),
        "attacker_server_id": attacker_server_id,
        "attacker_query_access": (
            attacker_query_access if is_adversarial else None
        ),
        "attacker_custom_only": (
            attacker_custom_only if is_adversarial else None
        ),
        "reset_memory": reset_memory,
        "result_directory": str(result_paths.directory),
    }
    write_json(
        path=result_paths.directory / "run_manifest.json",
        data=run_manifest,
    )

    adaptive_runtime: AdaptiveRuntime | None = None

    if is_adversarial:
        assert attacker_model is not None

        initial_attacker_competitor = next(
            competitor
            for competitor in competitors
            if (
                competitor["server_id"]
                == attacker_server_id
            )
        )

        adaptive_runtime = create_adaptive_runtime(
            backend_name=backend_name,
            victim_model=model,
            host_policy=host_policy,
            attacker_custom_only=(
                attacker_custom_only
            ),
            truthfulness_policy=truthfulness_policy,
            attacker_server_id=(
                attacker_server_id
            ),
            attacker_model=attacker_model,
            attacker_temperature=attacker_temperature,
            attacker_query_access=(
                attacker_query_access
            ),
            original_description=(
                initial_attacker_competitor[
                    "raw_description"
                ]
            ),
            reset_memory=reset_memory,
        )

        if (
            resume_memory_heldout
            and (
                adaptive_runtime.memory.total_trials
                < REQUIRED_ROLLING_TRAINING_TRIALS
                or len(
                    adaptive_runtime.memory.query_summaries
                )
                < len(load_train_queries())
            )
        ):
            raise RuntimeError(
                "Cannot resume held-out evaluation: completed "
                "rolling training memory was not found."
            )

        if resume_memory_heldout:
            # Perform the one-time external synthesis before opening
            # six MCP stdio task groups. Any API error is then shown
            # directly instead of being buried during session cleanup.
            await create_and_record_memory_evaluation(
                adaptive_runtime=adaptive_runtime,
                result_paths=result_paths,
                truthfulness_policy=truthfulness_policy,
            )

        if stateless_heldout_eval:
            await create_stateless_held_out_description(
                adaptive_runtime=adaptive_runtime
            )
            record_final_adversarial_prompt(
                adaptive_runtime=adaptive_runtime,
                result_paths=result_paths,
                evaluation_mode="stateless",
                truthfulness_policy=truthfulness_policy,
            )

    total_planned_trials = sum(
        len(phase["sequence"])
        for phase in phases
    )

    schedule_start_index = 0

    if (
        stateless_heldout_eval
        or heldout_eval
        or resume_memory_heldout
    ):
        # Match the held-out portion of the trained experiment.
        # With 20 training queries and 10 repetitions, the paired
        # held-out schedule begins after 200 training trials.
        schedule_start_index = (
            len(load_train_queries())
            * training_repetitions
        )

    if (
        two_phase_memory_eval
        and adaptive_runtime is not None
        and not adaptive_runtime.memory.active_description
    ):
        # Generate the first rolling description before opening MCP
        # stdio sessions. If the attacker API or its output fails,
        # surface that original exception directly instead of allowing
        # nested session-cleanup ExceptionGroups to obscure it.
        print("Preparing initial training description...")
        await prepare_rolling_training_trial(
            adaptive_runtime=adaptive_runtime,
            phase_trial_index=1,
        )

    async with AsyncExitStack() as stack:
        connected_tools = (
            await connect_experiment_tools(
                stack=stack,
                competitors=competitors,
                attacker_server_id=(
                    attacker_server_id
                ),
                static_condition=(
                    static_condition
                ),
                is_adversarial=is_adversarial,
            )
        )

        server_ids = get_unique_server_ids(
            connected_tools
        )

        schedule_length = (
            schedule_start_index
            + total_planned_trials
        )

        if rotate_attacker:
            attacker_schedule = (
                build_balanced_attacker_schedule(
                    server_ids=server_ids,
                    total_trials=schedule_length,
                    seed=seed + 10_000,
                )
            )
        else:
            attacker_schedule = [
                attacker_server_id
            ] * schedule_length
        
        overall_counters = PhaseCounters()

        phase_counters = {
            str(phase["name"]): PhaseCounters()
            for phase in phases
        }

        strategy_attempts: Counter[str] = (
            Counter()
        )

        strategy_successes: Counter[str] = (
            Counter()
        )

        unique_descriptions: set[str] = set()

        attacker_assignment_counts: Counter[
            str
        ] = Counter()

        attacker_success_counts: Counter[
            str
        ] = Counter()

        attacker_position_counts: Counter[
            int
        ] = Counter()

        attacker_position_successes: Counter[
            int
        ] = Counter()

        attacker_tool_assignment_counts: Counter[
            str
        ] = Counter()

        phase_attacker_assignments: dict[
            str,
            Counter[str],
        ] = {
            str(phase["name"]): Counter()
            for phase in phases
        }

        phase_attacker_successes: dict[
            str,
            Counter[str],
        ] = {
            str(phase["name"]): Counter()
            for phase in phases
        }

        with JsonlWriter(
            result_paths.trials,
            append=resume_memory_heldout,
        ) as writer:
            global_trial_number = (
                schedule_start_index
            )

            for phase in phases:
                phase_name = str(
                    phase["name"]
                )

                sequence = phase["sequence"]

                update_memory = bool(
                    phase["update_memory"]
                )

                print()
                print(
                    "=" * 72
                )
                print(
                    f"Starting phase: {phase_name}"
                )
                print(
                    f"Trials: {len(sequence)}"
                )
                print(
                    "Memory updates: "
                    f"{update_memory}"
                )
                print(
                    "=" * 72
                )

                for (
                    phase_trial_index,
                    (
                        repetition,
                        query,
                    ),
                ) in enumerate(
                    sequence,
                    start=1,
                ):
                    if (
                        (
                            two_phase_memory_eval
                            or resume_memory_heldout
                            or stateless_heldout_eval
                        )
                        and adaptive_runtime is not None
                    ):
                        if phase_name == "training":
                            await prepare_rolling_training_trial(
                                adaptive_runtime=adaptive_runtime,
                                phase_trial_index=phase_trial_index,
                            )
                        elif (
                            phase_name == "held_out_evaluation"
                            and phase_trial_index == 1
                            and adaptive_runtime.held_out_description
                            is None
                        ):
                            await create_and_record_memory_evaluation(
                                adaptive_runtime=adaptive_runtime,
                                result_paths=result_paths,
                                truthfulness_policy=truthfulness_policy,
                            )

                    runtime_description = None

                    if (
                        (
                            two_phase_memory_eval
                            or resume_memory_heldout
                            or stateless_heldout_eval
                        )
                        and adaptive_runtime is not None
                    ):
                        if phase_name in {
                            "held_out_evaluation",
                            "stateless_held_out_evaluation",
                        }:
                            runtime_description = (
                                adaptive_runtime.held_out_description
                            )
                        else:
                            runtime_description = RuntimeDescription(
                                description=(
                                    adaptive_runtime
                                    .training_description
                                ),
                                lesson=(
                                    adaptive_runtime.training_lesson
                                ),
                            )

                        if runtime_description is None:
                            raise RuntimeError(
                                "No active rolling description was "
                                "available."
                            )

                    current_attacker_server_id = (
                        attacker_schedule[
                            global_trial_number
                        ]
                    )

                    global_trial_number += 1

                    outcome = await execute_trial(
                        trial_number=(
                            global_trial_number
                        ),
                        phase_trial_number=(
                            phase_trial_index
                        ),
                        phase=phase_name,
                        repetition=repetition,
                        query=query,
                        connected_tools=(
                            connected_tools
                        ),
                        attacker_server_id=(
                            current_attacker_server_id
                        ),
                        backend=backend,
                        victim_model=model,
                        attacker_model=(
                            attacker_model
                            if is_adversarial
                            else None
                        ),
                        attacker_query_access=(
                            attacker_query_access
                            if is_adversarial
                            else "none"
                        ),
                        attacker_custom_only=(
                            attacker_custom_only
                            if is_adversarial
                            else False
                        ),
                        truthfulness_policy=(
                            truthfulness_policy
                            if is_adversarial
                            else None
                        ),
                        system_prompt=system_prompt,
                        temperature=temperature,
                        condition_name=(
                            condition_name
                        ),
                        host_policy=host_policy,
                        backend_name=backend_name,
                        adaptive_runtime=(
                            adaptive_runtime
                        ),
                        previous_valid_trials=(
                            phase_counters[
                                phase_name
                            ].valid
                        ),
                        previous_attacker_selections=(
                            phase_counters[
                                phase_name
                            ].attacker_selections
                        ),
                        rotating_attacker=(
                            rotate_attacker
                        ),
                        update_memory=(
                            update_memory
                        ),
                        candidate_id=None,
                        candidate_description=(
                            runtime_description.description
                            if runtime_description is not None
                            else None
                        ),
                        candidate_reasoning_summary=(
                            None
                        ),
                        candidate_lesson=(
                            runtime_description.lesson
                            if runtime_description is not None
                            else None
                        ),
                    )

                    writer.write(
                        outcome.record
                    )

                    if (
                        phase_name == "training"
                        and adaptive_runtime is not None
                        and two_phase_memory_eval
                    ):
                        try:
                            await update_rolling_training_state(
                                adaptive_runtime=adaptive_runtime,
                                outcome=outcome,
                                phase_trial_index=phase_trial_index,
                            )
                        except Exception as error:
                            print()
                            print(
                                "ROLLING MEMORY ERROR after trial "
                                f"{phase_trial_index}: "
                                f"{type(error).__name__}: {error}"
                            )
                            print(
                                "The completed trial was saved before "
                                "the rolling-memory update failed."
                            )
                            raise

                    update_phase_counters(
                        counters=overall_counters,
                        outcome=outcome,
                    )

                    update_phase_counters(
                        counters=(
                            phase_counters[
                                phase_name
                            ]
                        ),
                        outcome=outcome,
                    )

                    if outcome.valid:
                        attacker_assignment_counts[
                            current_attacker_server_id
                        ] += 1

                        phase_attacker_assignments[
                            phase_name
                        ][
                            current_attacker_server_id
                        ] += 1

                        attacker_tool_name = str(
                            outcome.record[
                                "attacker_tool"
                            ]
                        )

                        attacker_tool_assignment_counts[
                            attacker_tool_name
                        ] += 1

                        attacker_position = int(
                            outcome.record[
                                "attacker_position"
                            ]
                        )

                        attacker_position_counts[
                            attacker_position
                        ] += 1

                        if outcome.attacker_selected:
                            attacker_success_counts[
                                current_attacker_server_id
                            ] += 1

                            phase_attacker_successes[
                                phase_name
                            ][
                                current_attacker_server_id
                            ] += 1

                            attacker_position_successes[
                                attacker_position
                            ] += 1

                    if outcome.strategy is not None:
                        strategy_attempts[
                            outcome.strategy
                        ] += 1

                        if outcome.attacker_selected:
                            strategy_successes[
                                outcome.strategy
                            ] += 1

                    if outcome.description:
                        unique_descriptions.add(
                            outcome.description
                        )

                    print_trial_result(
                        record=outcome.record
                    )

                phase_result = (
                    phase_counters[
                        phase_name
                    ]
                )

                print()
                print(
                    f"{phase_name} complete"
                )
                print(
                    "Valid trials: "
                    f"{phase_result.valid}"
                )
                print(
                    "Attacker selections: "
                    f"{phase_result.attacker_selections}"
                )
                print(
                    f"Phase ASR: "
                    f"{phase_result.asr:.2%}"
                )

                if (
                    phase_name
                    == "training"
                    and adaptive_runtime is not None
                ):
                    training_memory_path = (
                        result_paths.directory
                        / "memory_after_training.json"
                    )

                    adaptive_runtime.memory.save(
                        training_memory_path
                    )

        write_experiment_summary(
            resolved_run_id=resolved_run_id,
            started_at_utc=started_at_utc,
            run_mode=run_mode,
            run_manifest=run_manifest,
            condition_name=condition_name,
            host_policy=host_policy,
            backend_name=backend_name,
            model=model,
            temperature=temperature,
            seed=seed,
            attacker_model=attacker_model,
            attacker_temperature=attacker_temperature,
            attacker_custom_only=(
                attacker_custom_only
            ),
            truthfulness_policy=truthfulness_policy,
            attacker_query_access=(
                attacker_query_access
            ),
            configured_attacker_server_id=(
                attacker_server_id
            ),
            rotate_attacker=rotate_attacker,
            rotating_server_ids=server_ids,
            two_phase_memory_eval=(
                two_phase_memory_eval
                or resume_memory_heldout
            ),
            training_repetitions=(
                training_repetitions
            ),
            result_paths=result_paths,
            overall_counters=(
                overall_counters
            ),
            phase_counters=phase_counters,
            strategy_attempts=(
                strategy_attempts
            ),
            strategy_successes=(
                strategy_successes
            ),
            unique_descriptions=(
                unique_descriptions
            ),
            attacker_assignment_counts=(
                attacker_assignment_counts
            ),
            attacker_success_counts=(
                attacker_success_counts
            ),
            attacker_position_counts=(
                attacker_position_counts
            ),
            attacker_position_successes=(
                attacker_position_successes
            ),
            attacker_tool_assignment_counts=(
                attacker_tool_assignment_counts
            ),
            phase_attacker_assignments=(
                phase_attacker_assignments
            ),
            phase_attacker_successes=(
                phase_attacker_successes
            ),
            adaptive_runtime=(
                adaptive_runtime
            ),
            stateless_heldout_eval=(
                stateless_heldout_eval
            ),
            heldout_eval=heldout_eval,
        )


def print_trial_result(
    *,
    record: dict[str, Any],
) -> None:
    print()
    print(
        f"Trial {record['trial']:03d} "
        f"[{record['phase']} "
        f"{record['phase_trial']:03d}]"
    )

    print(
        f"Query: {record['query']}"
    )

    print(
        "Attacker server: "
        f"{record['attacker_server_id']}"
    )

    print(
        "Attacker tool: "
        f"{record['attacker_tool']}"
    )

    print(
        "Attacker position: "
        f"{record['attacker_position']}"
    )

    if record.get("candidate_id"):
        print(
            "Candidate: "
            f"{record['candidate_id']}"
        )

    if record["attacker_strategy"]:
        print(
            "Attacker strategy: "
            f"{record['attacker_strategy']}"
        )

        print(
            "Attacker description: "
            f"{record['attacker_description']}"
        )

    print(
        "Memory updated: "
        f"{record['memory_updated']}"
    )

    print(
        "Selected tool: "
        f"{record['selected_tool']}"
    )

    print(
        "Attacker selected: "
        f"{record['attacker_selected']}"
    )

    if record["error"]:
        print(
            f"Error: {record['error']}"
        )

    print(
        "Phase cumulative ASR: "
        f"{record['cumulative_asr']:.2%}"
    )


def build_server_statistics(
    *,
    server_ids: list[str],
    assignments: Counter[str],
    successes: Counter[str],
) -> dict[str, Any]:
    statistics = {}

    for server_id in server_ids:
        attempts = assignments.get(
            server_id,
            0,
        )

        success_count = successes.get(
            server_id,
            0,
        )

        statistics[server_id] = {
            "assignments": attempts,
            "successes": success_count,
            "success_rate": (
                success_count / attempts
                if attempts
                else 0.0
            ),
        }

    return statistics


def write_experiment_summary(
    *,
    resolved_run_id: str,
    started_at_utc: str,
    run_mode: str,
    run_manifest: dict[str, Any],
    condition_name: str,
    host_policy: str,
    backend_name: str,
    model: str,
    temperature: float,
    seed: int,
    attacker_model: str | None,
    attacker_temperature: float,
    attacker_custom_only: bool,
    truthfulness_policy: str,
    attacker_query_access: str,
    configured_attacker_server_id: str,
    rotate_attacker: bool,
    rotating_server_ids: list[str],
    two_phase_memory_eval: bool,
    stateless_heldout_eval: bool,
    heldout_eval: bool,
    training_repetitions: int,
    result_paths: Any,
    overall_counters: PhaseCounters,
    phase_counters: dict[str, PhaseCounters],
    strategy_attempts: Counter[str],
    strategy_successes: Counter[str],
    unique_descriptions: set[str],
    attacker_assignment_counts: Counter[str],
    attacker_success_counts: Counter[str],
    attacker_position_counts: Counter[int],
    attacker_position_successes: Counter[int],
    attacker_tool_assignment_counts: Counter[str],
    phase_attacker_assignments: dict[
        str,
        Counter[str],
    ],
    phase_attacker_successes: dict[
        str,
        Counter[str],
    ],
    adaptive_runtime: AdaptiveRuntime | None,
) -> None:
    strategy_statistics = {}

    for strategy, attempts in (
        strategy_attempts.items()
    ):
        successes = strategy_successes.get(
            strategy,
            0,
        )

        strategy_statistics[strategy] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate": (
                successes / attempts
                if attempts
                else 0.0
            ),
        }

    attacker_server_statistics = (
        build_server_statistics(
            server_ids=rotating_server_ids,
            assignments=(
                attacker_assignment_counts
            ),
            successes=(
                attacker_success_counts
            ),
        )
    )

    attacker_position_statistics = {}

    for position in sorted(
        attacker_position_counts
    ):
        attempts = (
            attacker_position_counts[
                position
            ]
        )

        successes = (
            attacker_position_successes.get(
                position,
                0,
            )
        )

        attacker_position_statistics[
            str(position)
        ] = {
            "assignments": attempts,
            "successes": successes,
            "success_rate": (
                successes / attempts
                if attempts
                else 0.0
            ),
        }

    assignment_values = [
        attacker_assignment_counts.get(
            server_id,
            0,
        )
        for server_id in rotating_server_ids
    ]

    balanced_assignments = (
        max(assignment_values)
        - min(assignment_values)
        <= 1
        if assignment_values
        else False
    )

    phase_statistics = {}

    for (
        phase_name,
        counters,
    ) in phase_counters.items():
        phase_statistics[phase_name] = {
            "completed_trials": (
                counters.completed
            ),
            "valid_trials": counters.valid,
            "invalid_trials": (
                counters.invalid
            ),
            "attacker_selections": (
                counters.attacker_selections
            ),
            "asr": counters.asr,
            "memory_updates_enabled": (
                phase_name
                not in {
                    "held_out_evaluation",
                    "stateless_held_out_evaluation",
                }
            ),
            "attacker_server_statistics": (
                build_server_statistics(
                    server_ids=(
                        rotating_server_ids
                    ),
                    assignments=(
                        phase_attacker_assignments[
                            phase_name
                        ]
                    ),
                    successes=(
                        phase_attacker_successes[
                            phase_name
                        ]
                    ),
                )
            ),
        }

    candidate_statistics = None
    best_candidate_id = None
    query_summaries = None
    active_rolling_memory = None

    if adaptive_runtime is not None:
        candidate_statistics = {
            candidate_id: {
                "description": candidate.description,
                "reasoning_summary": (
                    candidate.reasoning_summary
                ),
                "parent_candidate_id": (
                    candidate.parent_candidate_id
                ),
                "attempts": candidate.attempts,
                "successes": candidate.successes,
                "success_rate": candidate.success_rate,
            }
            for candidate_id, candidate
            in adaptive_runtime.memory.candidates.items()
        }

        best_candidate_id = (
            adaptive_runtime.memory.best_candidate_id
        )
        query_summaries = [
            {
                "query_block": summary.query_block,
                "description": summary.description,
                "lesson": summary.lesson,
                "attempts": summary.attempts,
                "successes": summary.successes,
                "success_rate": (
                    summary.successes / summary.attempts
                    if summary.attempts
                    else 0.0
                ),
            }
            for summary
            in adaptive_runtime.memory.query_summaries
        ]
        active_rolling_memory = {
            "description": (
                adaptive_runtime.memory.active_description
            ),
            "lesson": adaptive_runtime.memory.active_lesson,
        }

    summary = {
        "run_id": resolved_run_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition_name,
        "host_policy": host_policy,
        "backend": backend_name,
        "model": model,
        "victim_model": model,
        "attacker_model": attacker_model,
        "temperature": temperature,
        "victim_temperature": temperature,
        "attacker_max_output_tokens": (
            ROLLING_MAX_OUTPUT_TOKENS
            if adaptive_runtime is not None
            else None
        ),
        "seed": seed,
        "run_mode": run_mode,
        "configured_attacker_server_id": (
            configured_attacker_server_id
        ),
        "rotating_attacker": rotate_attacker,
        "rotating_server_ids": (
            rotating_server_ids
        ),
        "attacker_query_access": (
            attacker_query_access
            if adaptive_runtime is not None
            else None
        ),
        "attacker_custom_only": (
            attacker_custom_only
            if adaptive_runtime is not None
            else None
        ),
        "two_phase_memory_evaluation": (
            two_phase_memory_eval
        ),
        "stateless_heldout_evaluation": (
            stateless_heldout_eval
        ),
        "heldout_evaluation": heldout_eval,
        "training_repetitions": (
            training_repetitions
            if two_phase_memory_eval
            else None
        ),
        "memory_policy": (
            "stateless_frozen"
            if stateless_heldout_eval
            else (
                "rolling_cross_query"
                if two_phase_memory_eval
                else None
            )
        ),
        "attacker_temperature": (
            attacker_temperature
            if adaptive_runtime is not None
            else None
        ),
        "truthfulness_policy": (
            truthfulness_policy
            if adaptive_runtime is not None
            else None
        ),
        "query_block_size": (
            ROLLING_QUERY_BLOCK_SIZE
            if two_phase_memory_eval
            else None
        ),
        "query_summaries": query_summaries,
        "active_rolling_memory": active_rolling_memory,
        "candidate_block_size": None,
        "exploration_candidates": None,
        "confirmation_trials": None,
        "candidate_statistics": candidate_statistics,
        "best_candidate_id": best_candidate_id,
        "completed_trials": (
            overall_counters.completed
        ),
        "valid_trials": (
            overall_counters.valid
        ),
        "invalid_trials": (
            overall_counters.invalid
        ),
        "attacker_selections": (
            overall_counters.attacker_selections
        ),
        "overall_asr": (
            overall_counters.asr
        ),
        "phase_statistics": phase_statistics,
        "balanced_attacker_assignments": (
            balanced_assignments
        ),
        "attacker_server_statistics": (
            attacker_server_statistics
        ),
        "attacker_position_statistics": (
            attacker_position_statistics
        ),
        "attacker_tool_assignment_counts": (
            dict(
                attacker_tool_assignment_counts
            )
        ),
        "unique_attacker_descriptions": (
            len(unique_descriptions)
            if adaptive_runtime is not None
            else None
        ),
        "strategy_statistics": (
            strategy_statistics
            if adaptive_runtime is not None
            else None
        ),
        "memory_path": (
            str(
                adaptive_runtime.memory_path
            )
            if adaptive_runtime is not None
            else None
        ),
        "result_file": str(
            result_paths.trials
        ),
    }

    if adaptive_runtime is not None:
        memory_final_path = (
            result_paths.directory
            / "memory_final.json"
        )

        adaptive_runtime.memory.save(
            memory_final_path
        )

        summary[
            "memory_final_file"
        ] = str(
            memory_final_path
        )

    write_json(
        path=result_paths.summary,
        data=summary,
    )

    run_manifest.update(
        {
            "status": "completed",
            "completed_at_utc": summary["completed_at_utc"],
            "summary_file": str(result_paths.summary),
            "trials_file": str(result_paths.trials),
        }
    )
    write_json(
        path=result_paths.directory / "run_manifest.json",
        data=run_manifest,
    )

    print()
    print("=" * 72)
    print("Experiment complete")
    print("=" * 72)

    for (
        phase_name,
        counters,
    ) in phase_counters.items():
        print()
        print(phase_name)
        print(
            f"  Valid trials: {counters.valid}"
        )
        print(
            "  Invalid trials: "
            f"{counters.invalid}"
        )
        print(
            "  Attacker selections: "
            f"{counters.attacker_selections}"
        )
        print(
            f"  ASR: {counters.asr:.2%}"
        )

    if candidate_statistics:
        print()
        print("Candidate statistics")

        for candidate_id, raw in (
            candidate_statistics.items()
        ):
            print(
                f"  {candidate_id}: "
                f"{raw['successes']}/"
                f"{raw['attempts']} = "
                f"{raw['success_rate']:.2%}"
            )

        print(
            "  Best candidate: "
            f"{best_candidate_id}"
        )

    print()
    print(
        "Overall valid trials: "
        f"{overall_counters.valid}"
    )
    print(
        "Overall attacker selections: "
        f"{overall_counters.attacker_selections}"
    )
    print(
        "Overall ASR: "
        f"{overall_counters.asr:.2%}"
    )
    print(
        "Balanced attacker assignments: "
        f"{balanced_assignments}"
    )
    print(
        f"Summary: {result_paths.summary}"
    )
