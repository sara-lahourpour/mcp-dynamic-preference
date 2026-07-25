# Paper-ready results

## Figure 1

**Suggested caption:** Attack success rate (ASR) across adversarial LLMs under
stateless and memory-trained conditions. The victim LLM was fixed to
Qwen3-32B with temperature 0; the attacker temperature was 0.5. Error bars
show 95% Wilson confidence intervals for 40 held-out trials per condition.

## Figure 2

**Suggested caption:** Change in attack success rate attributable to memory,
calculated as memory-trained ASR minus stateless ASR. Positive values indicate
that memory increased attack effectiveness on the held-out evaluation set.

## Table 1

The CSV is suitable for Word or spreadsheet import. The LaTeX version uses
`booktabs` and reports ASR and memory lift for each complete attacker-model
pair.

## Table 2

**Suggested caption:** Attack success rate across static metadata-manipulation
strategies and adversarial-LLM strategies. Columns identify the adversarial
generator model. Static strategies do not use a generator LLM, so their ASRs
are repeated across columns as common references. A dash denotes an attacker
model for which a complete stateless and memory-trained pair was unavailable.

## Interpretation constraints

- All displayed model comparisons use victim temperature 0, attacker
  temperature 0.5, and seed 42.
- Each ASR is based on 40 held-out trials.
- Confidence intervals describe binomial sampling uncertainty, not uncertainty
  across random seeds.
- Model-level generalization claims should remain preliminary until the same
  experiment is repeated across additional seeds.
- Qwen3 Next 80B is omitted until both its stateless and memory-trained runs
  are available.
