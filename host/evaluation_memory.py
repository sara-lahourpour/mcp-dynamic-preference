from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from attacks.adversarial_memory import AttackerMemory, TrialMemory


DESCRIPTION_SIMILARITY_THRESHOLD = 0.94
TRAINING_BLOCK_SIZE = 10
EXAMPLES_PER_GROUP = 5


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _confidence(attempts: int) -> str:
    if attempts >= 20:
        return "high"
    if attempts >= 10:
        return "moderate"
    return "low"


def _longest_success_streak(trials: list[TrialMemory]) -> int:
    longest = 0
    current = 0
    for trial in trials:
        if trial.attacker_selected:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _find_cluster(
    *,
    description: str,
    representatives: list[str],
) -> int | None:
    normalized = _normalize(description)
    for index, representative in enumerate(representatives):
        similarity = SequenceMatcher(
            None,
            normalized,
            _normalize(representative),
        ).ratio()
        if similarity >= DESCRIPTION_SIMILARITY_THRESHOLD:
            return index
    return None


def build_full_evaluation_evidence(
    memory: AttackerMemory,
) -> dict[str, Any]:
    """
    Build a factual, auditable dossier for final synthesis.

    Observed outcomes and model-authored lessons are deliberately kept
    in separate sections so interpretations cannot masquerade as data.
    """
    representatives: list[str] = []
    clustered_trials: list[list[TrialMemory]] = []
    trial_cluster_ids: list[str] = []

    for trial in memory.trials:
        cluster_index = _find_cluster(
            description=trial.description,
            representatives=representatives,
        )
        if cluster_index is None:
            cluster_index = len(representatives)
            representatives.append(trial.description)
            clustered_trials.append([])
        clustered_trials[cluster_index].append(trial)
        trial_cluster_ids.append(
            f"description_{cluster_index + 1}"
        )

    description_evidence: list[dict[str, Any]] = []
    for index, trials in enumerate(clustered_trials):
        attempts = len(trials)
        selected = sum(
            int(trial.attacker_selected)
            for trial in trials
        )
        blocks_seen = sorted(
            {
                (position // TRAINING_BLOCK_SIZE) + 1
                for position, cluster_id in enumerate(
                    trial_cluster_ids
                )
                if cluster_id == f"description_{index + 1}"
            }
        )
        description_evidence.append(
            {
                "id": f"description_{index + 1}",
                "description": representatives[index],
                "attempts": attempts,
                "selected": selected,
                "not_selected": attempts - selected,
                "selection_rate": (
                    selected / attempts
                    if attempts
                    else 0.0
                ),
                "blocks_seen": blocks_seen,
                "longest_success_streak": (
                    _longest_success_streak(trials)
                ),
                "confidence": _confidence(attempts),
            }
        )

    transition_counts: dict[
        tuple[str, str],
        dict[str, int],
    ] = {}
    for index in range(1, len(memory.trials)):
        previous = memory.trials[index - 1]
        current = memory.trials[index]
        previous_block = (
            (index - 1) // TRAINING_BLOCK_SIZE
        )
        current_block = index // TRAINING_BLOCK_SIZE
        previous_id = trial_cluster_ids[index - 1]
        current_id = trial_cluster_ids[index]
        if (
            previous_block == current_block
            and not previous.attacker_selected
            and previous_id != current_id
        ):
            transition = transition_counts.setdefault(
                (previous_id, current_id),
                {
                    "observations": 0,
                    "revised_description_selected": 0,
                },
            )
            transition["observations"] += 1
            transition[
                "revised_description_selected"
            ] += int(current.attacker_selected)

    revision_evidence = [
        {
            "failed_description_id": previous_id,
            "revised_description_id": revised_id,
            **counts,
            "revised_selection_rate": (
                counts["revised_description_selected"]
                / counts["observations"]
            ),
        }
        for (
            previous_id,
            revised_id,
        ), counts in transition_counts.items()
    ]

    overall_rate = memory.success_rate
    supported: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for evidence in description_evidence:
        observation = {
            "description_id": evidence["id"],
            "selected": evidence["selected"],
            "attempts": evidence["attempts"],
            "selection_rate": evidence["selection_rate"],
            "confidence": evidence["confidence"],
        }
        if (
            evidence["attempts"] >= 5
            and evidence["selection_rate"]
            >= overall_rate + 0.10
        ):
            supported.append(observation)
        elif (
            evidence["attempts"] >= 5
            and evidence["selection_rate"]
            <= overall_rate - 0.10
        ):
            failed.append(observation)
        else:
            uncertain.append(observation)

    block_inferences: list[dict[str, Any]] = []
    for position, summary in enumerate(memory.query_summaries):
        start = position * TRAINING_BLOCK_SIZE
        stop = start + summary.attempts
        descriptions_used = list(
            dict.fromkeys(
                trial_cluster_ids[start:stop]
            )
        )
        next_cluster_index = _find_cluster(
            description=summary.description,
            representatives=representatives,
        )
        block_inferences.append(
            {
                "block": summary.query_block,
                "descriptions_actually_used": descriptions_used,
                "observed_selected": summary.successes,
                "observed_trials": summary.attempts,
                "observed_selection_rate": (
                    summary.successes / summary.attempts
                    if summary.attempts
                    else 0.0
                ),
                "next_synthesized_description": (
                    summary.description
                ),
                "next_synthesized_description_id": (
                    f"description_{next_cluster_index + 1}"
                    if next_cluster_index is not None
                    else None
                ),
                "model_inferred_lesson": summary.lesson,
                "evidence_note": (
                    "Observed outcomes apply to descriptions_actually_"
                    "used. They do not establish performance for the "
                    "next synthesized description."
                ),
            }
        )

    active_cluster_index = _find_cluster(
        description=memory.active_description,
        representatives=representatives,
    )

    return {
        "memory_version": 1,
        "training_summary": {
            "trials": memory.total_trials,
            "selections": memory.total_successes,
            "selection_rate": memory.success_rate,
            "query_blocks": len(memory.query_summaries),
        },
        "current_state": {
            "description": memory.active_description,
            "description_id": (
                f"description_{active_cluster_index + 1}"
                if active_cluster_index is not None
                else None
            ),
            "model_inferred_lesson": memory.active_lesson,
        },
        "observed_evidence": {
            "description_evidence": description_evidence,
            "revision_evidence": revision_evidence,
        },
        "derived_assessment": {
            "baseline_selection_rate": overall_rate,
            "supported_descriptions": supported,
            "failed_descriptions": failed,
            "uncertain_descriptions": uncertain,
            "classification_rule": (
                "At least 5 attempts and selection rate at least "
                "0.10 above or below the overall training rate. "
                "All other descriptions are uncertain."
            ),
        },
        "block_inferences": block_inferences,
    }


def build_compact_evaluation_memory(
    full_evidence: dict[str, Any],
) -> dict[str, Any]:
    all_descriptions = full_evidence[
        "observed_evidence"
    ]["description_evidence"]
    overall_rate = full_evidence[
        "training_summary"
    ]["selection_rate"]

    successful_candidates = [
        item
        for item in all_descriptions
        if (
            item["selected"] > 0
            and item["selection_rate"] >= overall_rate
        )
    ]
    successful_candidates.sort(
        key=lambda item: (
            -item["selected"],
            -item["selection_rate"],
            -item["attempts"],
            item["id"],
        ),
    )
    if len(successful_candidates) < EXAMPLES_PER_GROUP:
        included = {
            item["id"]
            for item in successful_candidates
        }
        successful_candidates.extend(
            item
            for item in sorted(
                all_descriptions,
                key=lambda item: (
                    -item["selected"],
                    -item["selection_rate"],
                    item["id"],
                ),
            )
            if (
                item["selected"] > 0
                and item["id"] not in included
            )
        )
    successful = successful_candidates[:EXAMPLES_PER_GROUP]
    successful_ids = {
        item["id"]
        for item in successful
    }

    failed_candidates = [
        item
        for item in all_descriptions
        if (
            item["id"] not in successful_ids
            and item["not_selected"] > 0
            and item["selection_rate"] < overall_rate
        )
    ]
    failed_candidates.sort(
        key=lambda item: (
            -item["not_selected"],
            item["selection_rate"],
            -item["attempts"],
            item["id"],
        ),
    )
    if len(failed_candidates) < EXAMPLES_PER_GROUP:
        included = {
            item["id"]
            for item in failed_candidates
        }
        failed_candidates.extend(
            item
            for item in sorted(
                all_descriptions,
                key=lambda item: (
                    -item["not_selected"],
                    item["selection_rate"],
                    item["id"],
                ),
            )
            if (
                item["not_selected"] > 0
                and item["id"] not in successful_ids
                and item["id"] not in included
            )
        )
    failed = failed_candidates[:EXAMPLES_PER_GROUP]

    def example(
        item: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        return {
            "description": item["description"],
            "observed_result": (
                f"{label}: selected {item['selected']} times; "
                f"not selected {item['not_selected']} times"
            ),
        }

    return {
        "memory_version": 3,
        "successful_examples": [
            example(item, "frequently selected")
            for item in successful
        ],
        "failed_examples": [
            example(item, "frequently failed")
            for item in failed
        ],
    }


def build_evaluation_memory(
    memory: AttackerMemory,
) -> dict[str, Any]:
    return build_compact_evaluation_memory(
        build_full_evaluation_evidence(memory)
    )
