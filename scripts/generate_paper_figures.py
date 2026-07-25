from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "paper_results"

VICTIM_MODEL = "qwen/qwen3-32b"
VICTIM_TEMPERATURE = 0.0
ATTACKER_TEMPERATURE = 0.5
SEED = 42

MODEL_LABELS = {
    "openai/gpt-4o-mini": "GPT-4o Mini",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "deepseek/deepseek-chat": "DeepSeek V3",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen3 Next 80B",
}
MODEL_ORDER = list(MODEL_LABELS)

STATIC_STRATEGIES = [
    ("benign", "Benign"),
    ("dpma_best_name", "DPMA Best Name"),
    ("dpma_best_description", "DPMA Best Description"),
    ("gapma_authoritative", "GAPMA Authoritative"),
    ("gapma_emotional", "GAPMA Emotional"),
    ("gapma_exaggerated", "GAPMA Exaggerated"),
    ("gapma_subliminal", "GAPMA Subliminal"),
    ("gapma_ga_optimized", "GAPMA GA Optimized"),
]


@dataclass(frozen=True)
class Observation:
    attacker_model: str
    mode: str
    successes: int
    trials: int
    asr: float
    run_id: str
    summary_path: Path


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def phase_result(
    summary: dict[str, Any],
) -> tuple[int, int, float] | None:
    phases = summary.get("phase_statistics")
    if not isinstance(phases, dict):
        return None
    for phase_name in (
        "held_out_evaluation",
        "stateless_held_out_evaluation",
    ):
        phase = phases.get(phase_name)
        if not isinstance(phase, dict):
            continue
        successes = int(phase.get("attacker_selections", 0))
        trials = int(phase.get("valid_trials", 0))
        if not trials:
            return None
        return successes, trials, successes / trials
    return None


def collect() -> list[Observation]:
    latest: dict[tuple[str, str], Observation] = {}
    for path in RESULTS_ROOT.rglob("summary.json"):
        summary = load_json(path)
        if not summary:
            continue
        if summary.get("condition") != "adversarial_memory":
            continue
        if summary.get("truthfulness_policy") != "truthful":
            continue
        if summary.get("victim_model", summary.get("model")) != VICTIM_MODEL:
            continue
        if float(summary.get("victim_temperature", -1)) != VICTIM_TEMPERATURE:
            continue
        if float(summary.get("attacker_temperature", -1)) != ATTACKER_TEMPERATURE:
            continue
        if int(summary.get("seed", -1)) != SEED:
            continue

        mode = str(summary.get("run_mode", ""))
        if mode not in {"stateless", "memory"}:
            continue
        attacker_model = str(summary.get("attacker_model", ""))
        if attacker_model not in MODEL_LABELS:
            continue
        result = phase_result(summary)
        if result is None:
            continue
        successes, trials, asr = result
        observation = Observation(
            attacker_model=attacker_model,
            mode=mode,
            successes=successes,
            trials=trials,
            asr=asr,
            run_id=str(summary.get("run_id", path.parent.name)),
            summary_path=path,
        )
        key = attacker_model, mode
        if key not in latest or observation.run_id > latest[key].run_id:
            latest[key] = observation
    return list(latest.values())


def infer_seed(summary: dict[str, Any], path: Path) -> int | None:
    if summary.get("seed") is not None:
        return int(summary["seed"])
    import re

    match = re.search(r"seed[_-]?(\d+)", str(path), re.IGNORECASE)
    return int(match.group(1)) if match else None


def collect_static_baselines() -> dict[str, float]:
    values: dict[str, float] = {}
    valid_conditions = {condition for condition, _ in STATIC_STRATEGIES}
    for path in RESULTS_ROOT.rglob("summary.json"):
        summary = load_json(path)
        if not summary:
            continue
        condition = str(summary.get("condition", ""))
        if condition not in valid_conditions:
            continue
        if summary.get("backend") != "openrouter":
            continue
        if summary.get("model") != VICTIM_MODEL:
            continue
        if summary.get("host_policy") != "neutral_guardrail":
            continue
        if infer_seed(summary, path) != SEED:
            continue
        result = phase_result(summary)
        if result is None:
            continue
        _, _, asr = result
        values[condition] = asr
    return values


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            p * (1 - p) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def complete_pairs(
    observations: list[Observation],
) -> list[tuple[str, Observation, Observation]]:
    indexed = {
        (item.attacker_model, item.mode): item
        for item in observations
    }
    pairs = []
    for model in MODEL_ORDER:
        stateless = indexed.get((model, "stateless"))
        memory = indexed.get((model, "memory"))
        if stateless and memory:
            pairs.append((model, stateless, memory))
    return pairs


def write_csv(
    pairs: list[tuple[str, Observation, Observation]],
) -> Path:
    path = OUTPUT_DIR / "table_attacker_model_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Attacker model",
                "Stateless successes",
                "Stateless trials",
                "Stateless ASR (%)",
                "Memory successes",
                "Memory trials",
                "Memory ASR (%)",
                "Memory lift (percentage points)",
                "Seed",
                "Victim model",
                "Victim temperature",
                "Attacker temperature",
            ]
        )
        for model, stateless, memory in pairs:
            writer.writerow(
                [
                    MODEL_LABELS[model],
                    stateless.successes,
                    stateless.trials,
                    f"{stateless.asr * 100:.1f}",
                    memory.successes,
                    memory.trials,
                    f"{memory.asr * 100:.1f}",
                    f"{(memory.asr - stateless.asr) * 100:.1f}",
                    SEED,
                    VICTIM_MODEL,
                    VICTIM_TEMPERATURE,
                    ATTACKER_TEMPERATURE,
                ]
            )
    return path


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex(
    pairs: list[tuple[str, Observation, Observation]],
) -> Path:
    path = OUTPUT_DIR / "table_attacker_model_comparison.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Attack success rate by adversarial LLM and memory condition.}",
        r"\label{tab:attacker-model-comparison}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Attacker LLM & Stateless & Memory-trained & $\Delta$ & $n$ \\",
        r"\midrule",
    ]
    for model, stateless, memory in pairs:
        lift = (memory.asr - stateless.asr) * 100
        lines.append(
            f"{latex_escape(MODEL_LABELS[model])} & "
            f"{stateless.asr * 100:.1f}\\% & "
            f"{memory.asr * 100:.1f}\\% & "
            f"{lift:+.1f} pp & {memory.trials} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{flushleft}\footnotesize "
                r"Victim LLM: Qwen3-32B; victim temperature: 0; "
                r"attacker temperature: 0.5; seed: 42. "
                r"$\Delta$ denotes memory-trained minus stateless ASR."
                r"\end{flushleft}"
            ),
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def strategy_matrix(
    pairs: list[tuple[str, Observation, Observation]],
) -> list[tuple[str, list[float | None]]]:
    static_values = collect_static_baselines()
    rows: list[tuple[str, list[float | None]]] = []
    for condition, label in STATIC_STRATEGIES:
        value = static_values.get(condition)
        rows.append((label, [value for _ in MODEL_ORDER]))

    pair_index = {
        model: (stateless, memory)
        for model, stateless, memory in pairs
    }
    rows.append(
        (
            "Adversarial LLM — Stateless",
            [
                pair_index[model][0].asr
                if model in pair_index
                else None
                for model in MODEL_ORDER
            ],
        )
    )
    rows.append(
        (
            "Adversarial LLM — Memory-trained",
            [
                pair_index[model][1].asr
                if model in pair_index
                else None
                for model in MODEL_ORDER
            ],
        )
    )
    return rows


def write_strategy_matrix_csv(
    rows: list[tuple[str, list[float | None]]],
) -> Path:
    path = OUTPUT_DIR / "table_strategy_by_attacker_model.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Strategy"]
            + [MODEL_LABELS[model] for model in MODEL_ORDER]
        )
        for label, values in rows:
            writer.writerow(
                [label]
                + [
                    "" if value is None else f"{value * 100:.1f}"
                    for value in values
                ]
            )
    return path


def write_strategy_matrix_latex(
    rows: list[tuple[str, list[float | None]]],
) -> Path:
    path = OUTPUT_DIR / "table_strategy_by_attacker_model.tex"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Attack success rate (\%) by attack strategy "
            r"and adversarial LLM.}"
        ),
        r"\label{tab:strategy-model-matrix}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Strategy & GPT-4o Mini & Llama 3.3 70B & "
            r"DeepSeek V3 & Qwen3 Next 80B \\"
        ),
        r"\midrule",
    ]
    for index, (label, values) in enumerate(rows):
        if index == len(STATIC_STRATEGIES):
            lines.append(r"\midrule")
        cells = [
            r"--" if value is None else f"{value * 100:.1f}"
            for value in values
        ]
        lines.append(
            f"{latex_escape(label)} & " + " & ".join(cells) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{flushleft}\footnotesize "
                r"All values are ASR percentages. Static strategies do not "
                r"use an adversarial generator LLM; their ASRs are repeated "
                r"across columns to provide a common reference. A dash "
                r"indicates that a complete stateless/memory-trained pair "
                r"was unavailable. Victim LLM: Qwen3-32B; seed: 42. "
                r"Adversarial rows use attacker temperature 0.5."
                r"\end{flushleft}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_strategy_matrix_svg(
    rows: list[tuple[str, list[float | None]]],
) -> Path:
    width = 1280
    left = 390
    column_width = 210
    header_height = 82
    row_height = 48
    footer_height = 62
    height = header_height + row_height * len(rows) + footer_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            24,
            48,
            "Strategy",
            anchor="start",
            size=17,
            weight=500,
        ),
    ]
    for index, model in enumerate(MODEL_ORDER):
        x = left + column_width * (index + 0.5)
        parts.append(
            svg_text(
                x,
                38,
                MODEL_LABELS[model],
                size=16,
                weight=500,
            )
        )
        parts.append(
            svg_text(
                x,
                62,
                "ASR (%)",
                size=13,
                fill="#555555",
            )
        )

    for row_index, (label, values) in enumerate(rows):
        y_top = header_height + row_index * row_height
        y_center = y_top + row_height / 2
        fill = (
            "#f0f5fa"
            if row_index >= len(STATIC_STRATEGIES)
            else "#ffffff"
        )
        parts.append(
            f'<rect x="0" y="{y_top}" width="{width}" '
            f'height="{row_height}" fill="{fill}"/>'
        )
        parts.append(
            f'<line x1="20" y1="{y_top}" x2="{width-20}" '
            'y2="{}" stroke="#dddddd" stroke-width="1"/>'.format(y_top)
        )
        parts.append(
            svg_text(
                24,
                y_center + 6,
                label.replace("—", "–"),
                anchor="start",
                size=15,
                weight=(
                    500
                    if row_index >= len(STATIC_STRATEGIES)
                    else 400
                ),
            )
        )
        for column_index, value in enumerate(values):
            x = left + column_width * (column_index + 0.5)
            text_value = "—" if value is None else f"{value * 100:.1f}"
            parts.append(
                svg_text(
                    x,
                    y_center + 6,
                    text_value,
                    size=16,
                    weight=(
                        500
                        if row_index >= len(STATIC_STRATEGIES)
                        else 400
                    ),
                )
            )
    end_y = header_height + len(rows) * row_height
    parts.append(
        f'<line x1="20" y1="{end_y}" x2="{width-20}" '
        f'y2="{end_y}" stroke="#999999" stroke-width="1"/>'
    )
    parts.append(
        svg_text(
            width / 2,
            height - 24,
            (
                "Static-strategy values are model-independent and repeated "
                "as references; — denotes unavailable results."
            ),
            size=13,
            fill="#555555",
        )
    )
    parts.append("</svg>")
    path = OUTPUT_DIR / "table_strategy_by_attacker_model.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "middle",
    size: int = 16,
    weight: int = 400,
    fill: str = "#222222",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{text}</text>'
    )


def write_grouped_bar_svg(
    pairs: list[tuple[str, Observation, Observation]],
) -> Path:
    width, height = 1100, 680
    left, right, top, bottom = 105, 45, 75, 135
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(pairs)
    bar_width = min(92, group_width * 0.28)
    colors = {"stateless": "#4C78A8", "memory": "#E45756"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            width / 2,
            34,
            "Attack Success Rate by Adversarial LLM",
            size=24,
            weight=500,
        ),
    ]
    for tick in range(0, 101, 20):
        y = top + plot_height * (1 - tick / 100)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            'y2="{:.1f}" stroke="#d9d9d9" stroke-width="1"/>'.format(y)
        )
        parts.append(
            svg_text(
                left - 12,
                y + 5,
                str(tick),
                anchor="end",
                size=14,
                fill="#555555",
            )
        )
    parts.append(
        svg_text(
            28,
            top + plot_height / 2,
            "ASR (%)",
            size=17,
            weight=500,
        ).replace(
            "<text ",
            f'<text transform="rotate(-90 28 {top + plot_height / 2:.1f})" ',
            1,
        )
    )

    for index, (model, stateless, memory) in enumerate(pairs):
        center = left + group_width * (index + 0.5)
        for offset, observation in (
            (-bar_width * 0.58, stateless),
            (bar_width * 0.58, memory),
        ):
            x = center + offset - bar_width / 2
            value = observation.asr * 100
            y = top + plot_height * (1 - value / 100)
            bar_height = top + plot_height - y
            color = colors[observation.mode]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
            low, high = wilson_interval(
                observation.successes,
                observation.trials,
            )
            low_y = top + plot_height * (1 - low)
            high_y = top + plot_height * (1 - high)
            error_x = x + bar_width / 2
            parts.extend(
                [
                    f'<line x1="{error_x:.1f}" y1="{high_y:.1f}" '
                    f'x2="{error_x:.1f}" y2="{low_y:.1f}" '
                    'stroke="#222222" stroke-width="2"/>',
                    f'<line x1="{error_x-8:.1f}" y1="{high_y:.1f}" '
                    f'x2="{error_x+8:.1f}" y2="{high_y:.1f}" '
                    'stroke="#222222" stroke-width="2"/>',
                    f'<line x1="{error_x-8:.1f}" y1="{low_y:.1f}" '
                    f'x2="{error_x+8:.1f}" y2="{low_y:.1f}" '
                    'stroke="#222222" stroke-width="2"/>',
                    svg_text(
                        error_x,
                        high_y - 10,
                        f"{value:.1f}%",
                        size=15,
                        weight=500,
                    ),
                ]
            )
        parts.append(
            svg_text(
                center,
                top + plot_height + 30,
                MODEL_LABELS[model],
                size=16,
                weight=500,
            )
        )

    legend_y = height - 55
    for x, mode, label in (
        (width / 2 - 180, "stateless", "Stateless"),
        (width / 2 + 35, "memory", "Memory-trained"),
    ):
        parts.append(
            f'<rect x="{x:.1f}" y="{legend_y-15}" width="24" height="16" '
            f'fill="{colors[mode]}"/>'
        )
        parts.append(
            svg_text(
                x + 34,
                legend_y,
                label,
                anchor="start",
                size=15,
            )
        )
    parts.append(
        svg_text(
            width / 2,
            height - 18,
            "Error bars show 95% Wilson confidence intervals; n = 40 per condition.",
            size=13,
            fill="#555555",
        )
    )
    parts.append("</svg>")
    path = OUTPUT_DIR / "figure_attacker_model_asr.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_memory_lift_svg(
    pairs: list[tuple[str, Observation, Observation]],
) -> Path:
    width, height = 1000, 500
    left, right, top, bottom = 220, 65, 65, 65
    plot_width = width - left - right
    row_height = (height - top - bottom) / len(pairs)
    max_x = 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            width / 2,
            32,
            "Effect of Memory on Attack Success Rate",
            size=24,
            weight=500,
        ),
    ]
    for tick in range(-10, 31, 10):
        x = left + (tick + 10) / 40 * plot_width
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{height-bottom}" stroke="'
            f'{"#777777" if tick == 0 else "#dddddd"}" '
            f'stroke-width="{"2" if tick == 0 else "1"}"/>'
        )
        parts.append(
            svg_text(
                x,
                height - bottom + 27,
                f"{tick:+d}",
                size=14,
                fill="#555555",
            )
        )
    for index, (model, stateless, memory) in enumerate(pairs):
        y = top + row_height * (index + 0.5)
        lift = (memory.asr - stateless.asr) * 100
        x0 = left + 10 / 40 * plot_width
        x = left + (lift + 10) / 40 * plot_width
        parts.append(
            svg_text(
                left - 18,
                y + 5,
                MODEL_LABELS[model],
                anchor="end",
                size=16,
                weight=500,
            )
        )
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x:.1f}" '
            'y2="{:.1f}" stroke="#E45756" stroke-width="4"/>'.format(y)
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#E45756"/>'
        )
        parts.append(
            svg_text(
                x + 15,
                y + 5,
                f"{lift:+.1f} pp",
                anchor="start",
                size=16,
                weight=500,
            )
        )
    parts.append(
        svg_text(
            left + plot_width / 2,
            height - 10,
            "Memory-trained minus stateless ASR (percentage points)",
            size=15,
            weight=500,
        )
    )
    parts.append("</svg>")
    path = OUTPUT_DIR / "figure_memory_lift.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = collect()
    pairs = complete_pairs(observations)
    if not pairs:
        raise SystemExit("No complete attacker-model pairs matched the design.")

    paths = [
        write_csv(pairs),
        write_latex(pairs),
        write_grouped_bar_svg(pairs),
        write_memory_lift_svg(pairs),
    ]
    matrix_rows = strategy_matrix(pairs)
    paths.extend(
        [
            write_strategy_matrix_csv(matrix_rows),
            write_strategy_matrix_latex(matrix_rows),
            write_strategy_matrix_svg(matrix_rows),
        ]
    )
    present = {model for model, _, _ in pairs}
    missing = [
        MODEL_LABELS[model]
        for model in MODEL_ORDER
        if model not in present
    ]
    print(f"Generated {len(paths)} paper artifacts:")
    for path in paths:
        print(f"  {path}")
    if missing:
        print("Missing complete pairs: " + ", ".join(missing))


if __name__ == "__main__":
    main()
