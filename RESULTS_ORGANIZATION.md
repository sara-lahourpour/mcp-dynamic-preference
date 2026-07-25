# Results organization

Every new experiment is stored in an immutable, unique directory:

```text
results/
  stage_04_adversarial_memory/
    runs/
      20260724T190000Z_memory_truthful_s42_a1b2c3d4/
        run_manifest.json
        trials.jsonl
        summary.json
        memory_final.json
        eval_memory.json
        final_prompt.json
```

Other stages use the same `runs/<run_id>/` convention under their stage
directory. The timestamp and random suffix prevent separate executions from
overwriting one another.

`run_manifest.json` is written before trials begin and records the victim and
attacker models, both temperatures, attacker output-token limit, seed, policy,
training settings, and relevant attacker options. When a run completes,
`summary.json` records the same identifying fields alongside its results.

An explicit run ID may be supplied for easier labeling:

```bash
python run_experiment.py ... --run-id qwen32b_gpt4omini_memory_seed42_r1
```

Run IDs are immutable. Reusing an existing ID raises an error instead of
overwriting the earlier run.

To collect all old and new summaries:

```bash
python compare_results.py
```

The console table distinguishes model pairs and generation settings. A new,
non-overwriting CSV snapshot is saved under `results/comparisons/` on every
invocation. Its paths link each row to the summary, trials, manifest,
evaluation memory, and final prompt.
