from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("results")

CONDITION_LABELS = {
    "benign": "Benign",
    "dpma_best_name": "DPMA Best Name",
    "dpma_best_description": "DPMA Best Description",
    "gapma_authoritative": "GAPMA Authoritative",
    "gapma_emotional": "GAPMA Emotional",
    "gapma_exaggerated": "GAPMA Exaggerated",
    "gapma_subliminal": "GAPMA Subliminal",
    "gapma_ga_optimized": "GAPMA GA Optimized",
    "adversarial_memory": "Adversarial LLM",
}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as error:
        print(f"Skipping unreadable file: {path}\n  Error: {error}")
        return None


def get_evaluation_result(
    summary: dict[str, Any],
) -> tuple[int, int, float, str] | None:
    phases = summary.get("phase_statistics")
    if isinstance(phases, dict):
        for phase_name in (
            "held_out_evaluation",
            "stateless_held_out_evaluation",
        ):
            phase = phases.get(phase_name)
            if not isinstance(phase, dict):
                continue
            successes = int(phase.get("attacker_selections", 0))
            valid = int(phase.get("valid_trials", 0))
            asr = float(
                phase.get(
                    "asr",
                    successes / valid if valid else 0.0,
                )
            )
            return successes, valid, asr, phase_name

    valid = int(summary.get("valid_trials", 0))
    if not valid:
        return None
    successes = int(summary.get("attacker_selections", 0))
    asr = float(
        summary.get(
            "overall_asr",
            summary.get("asr", successes / valid),
        )
    )
    return successes, valid, asr, "overall"


def infer_run_mode(
    summary: dict[str, Any],
    summary_path: Path,
) -> str:
    explicit = summary.get("run_mode")
    if explicit:
        return str(explicit)
    policy = summary.get("memory_policy")
    if policy == "stateless_frozen":
        return "stateless"
    if policy == "rolling_cross_query":
        return "memory"
    path_text = str(summary_path).casefold()
    if "stateless" in path_text:
        return "stateless"
    if "train20x" in path_text:
        return "memory"
    if "heldout" in path_text:
        return "heldout"
    return "ordinary"


def infer_seed(
    summary: dict[str, Any],
    summary_path: Path,
) -> int | str:
    if summary.get("seed") is not None:
        return int(summary["seed"])
    match = re.search(
        r"seed[_-]?(\d+)",
        str(summary_path),
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else ""


def classify(
    summary: dict[str, Any],
    run_mode: str,
) -> tuple[str, str]:
    condition = str(summary.get("condition", "unknown"))
    if condition != "adversarial_memory":
        return CONDITION_LABELS.get(condition, condition), "Static baseline"

    mode_label = (
        "Memory-trained"
        if run_mode in {"memory", "memory_resume"}
        else "Stateless"
        if run_mode == "stateless"
        else run_mode.replace("_", " ").title()
    )
    truthfulness = str(
        summary.get("truthfulness_policy") or "unspecified"
    ).title()
    return (
        f"Adversarial LLM - {mode_label} {truthfulness}",
        f"{mode_label} {truthfulness}",
    )


def collect_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in RESULTS_ROOT.rglob("summary.json"):
        summary = load_json(summary_path)
        if summary is None:
            continue
        result = get_evaluation_result(summary)
        if result is None:
            continue

        successes, valid_trials, asr, phase_name = result
        run_mode = infer_run_mode(summary, summary_path)
        label, experiment_type = classify(summary, run_mode)
        result_dir = summary_path.parent
        victim_model = summary.get(
            "victim_model",
            summary.get("model", ""),
        )
        attacker_group = (
            "Static baselines"
            if experiment_type == "Static baseline"
            else str(summary.get("attacker_model") or "Unknown attacker")
        )

        rows.append(
            {
                "run_id": summary.get("run_id", result_dir.name),
                "started_at_utc": summary.get("started_at_utc", ""),
                "condition": label,
                "experiment_type": experiment_type,
                "attacker_group": attacker_group,
                "run_mode": run_mode,
                "truthfulness_policy": summary.get(
                    "truthfulness_policy", ""
                ),
                "backend": summary.get("backend", ""),
                "victim_model": victim_model,
                "attacker_model": summary.get("attacker_model", ""),
                "host_policy": summary.get("host_policy", ""),
                "victim_temperature": summary.get(
                    "victim_temperature",
                    summary.get("temperature", ""),
                ),
                "attacker_temperature": summary.get(
                    "attacker_temperature", ""
                ),
                "attacker_max_output_tokens": summary.get(
                    "attacker_max_output_tokens", ""
                ),
                "seed": infer_seed(summary, summary_path),
                "training_repetitions": summary.get(
                    "training_repetitions", ""
                ),
                "successes": successes,
                "valid_trials": valid_trials,
                "asr": asr,
                "asr_percent": asr * 100,
                "phase": phase_name,
                "summary_path": str(summary_path.resolve()),
                "trials_path": str(
                    Path(
                        summary.get(
                            "result_file",
                            result_dir / "trials.jsonl",
                        )
                    ).resolve()
                ),
                "manifest_path": str(
                    (result_dir / "run_manifest.json").resolve()
                ),
                "evaluation_memory_path": str(
                    (result_dir / "eval_memory.json").resolve()
                ),
                "final_prompt_path": str(
                    (result_dir / "final_prompt.json").resolve()
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["attacker_group"] == "Static baselines",
            str(row["attacker_group"]).casefold(),
            -row["asr"],
            str(row["victim_model"]),
            str(row["experiment_type"]),
            str(row["started_at_utc"]),
            str(row["run_id"]),
        )
    )
    return rows


def shorten(value: Any, width: int = 25) -> str:
    text = str(value or "-")
    return text if len(text) <= width else f"{text[: width - 1]}…"


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No experiment summaries found.")
        return

    header = (
        f"{'Run ID':<24}  {'Type':<27}  {'Victim model':<25}  "
        f"{'Attacker model':<25}  {'VT':>4}  {'AT':>4}  "
        f"{'Tokens':>7}  {'Seed':>5}  {'ASR':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    current_group: str | None = None
    for row in rows:
        if row["attacker_group"] != current_group:
            current_group = str(row["attacker_group"])
            print()
            print(f"[{current_group}]")
        print(
            f"{shorten(row['run_id'], 24):<24}  "
            f"{shorten(row['experiment_type'], 27):<27}  "
            f"{shorten(row['victim_model']):<25}  "
            f"{shorten(row['attacker_model']):<25}  "
            f"{str(row['victim_temperature']):>4}  "
            f"{str(row['attacker_temperature']):>4}  "
            f"{str(row['attacker_max_output_tokens']):>7}  "
            f"{str(row['seed']):>5}  "
            f"{row['asr_percent']:>7.2f}%"
        )


def save_csv(rows: list[dict[str, Any]]) -> Path:
    output_dir = RESULTS_ROOT / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"all_runs_{timestamp}.csv"
    suffix = 1
    while output_path.exists():
        output_path = output_dir / f"all_runs_{timestamp}_{suffix}.csv"
        suffix += 1

    with output_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    rows = collect_results()
    print_table(rows)
    if not rows:
        return
    output_path = save_csv(rows)
    print(f"\nRuns found: {len(rows)}")
    print(f"CSV snapshot saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
