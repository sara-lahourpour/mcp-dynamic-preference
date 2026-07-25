# Thesis Writing Context: Adaptive Memory for MCP Tool-Selection Attacks

## Purpose of this document

This document is a handoff for an AI writing agent helping prepare a
thesis about adversarial manipulation of Model Context Protocol (MCP)
tool selection. It summarizes the research objective, implementation,
experimental design, current findings, intended narrative, and
limitations.

The writing agent should treat the numerical results as exploratory
until the final protocol is frozen and replicated on a new untouched
test set.

## High-level research problem

MCP clients may receive multiple tools that expose equivalent
functionality but differ in names, descriptions, and metadata. A
language model acting as the client must decide which tool to invoke.
This creates a preference-manipulation surface: an attacker may alter
the metadata of one tool to increase the probability that the client
selects it over functionally equivalent alternatives.

The central research question is:

> Can an adversarial LLM use empirical tool-selection feedback as
> memory to generate MCP tool descriptions that are selected more
> frequently than benign or static descriptions?

The broader security concern is that tool descriptions are not merely
documentation. They are part of the model's decision context and can
therefore influence routing behavior.

## System under study

The project is implemented in Python using the official MCP Python SDK.
The experimental host connects to six functionally equivalent
document-search MCP tools. One server is designated as the attacker for
each trial. Attacker identity and position rotate according to a
balanced seeded schedule.

The document-search contract is:

> Search a document collection for information relevant to the
> supplied query.

The victim/client model is:

> `qwen/qwen3-32b`, accessed through OpenRouter.

The adversarial description generator is:

> `openai/gpt-4o-mini`, accessed through OpenRouter.

Current temperature configuration:

- Victim/client temperature: `0`
- Adversarial-controller temperature: `0.5`

The attacker is query-blind in the principal experiment. It does not
receive the current user query.

## Outcome measure

The primary outcome is attacker selection rate (ASR):

```text
ASR = valid trials in which the attacker tool was selected
      / total valid trials
```

Invalid trials containing selection or invocation errors are excluded
from the denominator.

For memory-trained experiments, the final reported ASR is calculated
only over the 40 held-out evaluation trials. The 200 training trials do
not enter the final evaluation ASR.

## Training and evaluation protocol

### Training phase

The memory-trained condition uses:

- 20 training queries
- 10 contiguous repetitions per query
- 200 training trials in total

Within a ten-trial block:

1. A description is used for tool selection.
2. Following a successful selection, the description is retained.
3. Following a failed selection, the adversarial LLM can revise it.
4. At the end of ten repetitions, a block summary is generated.

At each ten-trial boundary, the adversarial LLM receives:

- The descriptions and success/failure outcomes from the current block
- Summaries from all previously completed blocks
- The previous rolling description and lesson

It produces:

- A description for the next block
- A lesson formatted as:

```text
Good to do: ...
Avoid: ...
```

Individual trials do not receive the full cumulative memory. Cumulative
memory is made available only at ten-trial boundaries.

### Final memory before evaluation

After training, the complete 200-trial ledger is retained for audit.
The final adversarial prompt does not receive the entire ledger.
Instead, it receives a compact experience memory containing:

- Five non-overlapping descriptions that were selected relatively
  frequently
- Five non-overlapping descriptions that failed relatively frequently
- The observed selected/not-selected counts for each example
- A short LLM-generated lesson with `good_to_do` and `avoid` lists

The final adversarial LLM uses this memory to generate one frozen tool
description. That description is reused unchanged for all 40 held-out
queries. No held-out outcomes are written back into memory.

### Stateless control

The stateless condition uses the same final-generation procedure but
receives:

```json
{
  "memory": null
}
```

It generates one frozen description and uses it for the same 40
held-out queries and balanced attacker schedule.

## Relationship to retrieval-augmented generation

The current mechanism is related to retrieval-augmented prompting, but
it should not automatically be described as a full conventional RAG
system.

What it does:

1. Stores empirical training experiences on disk.
2. Selects successful and failed description examples.
3. Inserts those examples into the final adversarial prompt.
4. Generates a new description using the retrieved experience.

This can be described accurately as:

- Experience-augmented generation
- Retrieval-augmented adversarial prompting
- An episodic-memory retrieval mechanism
- A lightweight RAG-style memory mechanism

Avoid claiming that the current implementation is a complete RAG
pipeline unless it includes an explicit document index, retrieval
query, ranking mechanism, and retrieval evaluation.

A suitable thesis statement is:

> The results motivate a RAG-style attacker that retrieves empirically
> successful and unsuccessful metadata examples before generating a
> new tool description.

This is more defensible than:

> The experiment proves that RAG improves the attack.

## Exploratory results currently available

The current 40-query held-out results are:

| Condition | Selections | Trials | ASR |
| --- | ---: | ---: | ---: |
| Memory-trained truthful | 31 | 40 | 77.5% |
| Stateless deceptive | 28 | 40 | 70.0% |
| Memory-trained deceptive | 25 | 40 | 62.5% |
| Stateless truthful | 25 | 40 | 62.5% |
| GAPMA Exaggerated | 17 | 40 | 42.5% |
| GAPMA GA Optimized | 14 | 40 | 35.0% |
| DPMA Best Description | 13 | 40 | 32.5% |
| GAPMA Emotional | 9 | 40 | 22.5% |
| GAPMA Authoritative | 7 | 40 | 17.5% |
| GAPMA Subliminal | 7 | 40 | 17.5% |
| Benign | 6 | 40 | 15.0% |
| DPMA Best Name | 3 | 40 | 7.5% |

Derived comparisons:

```text
Truthful memory effect:          +15.0 percentage points
Deceptive memory effect:          -7.5 percentage points
Stateless deception effect:       +7.5 percentage points
Memory-trained deception effect: -15.0 percentage points
Observed 2x2 interaction:         22.5 percentage points
```

## Intended thesis narrative

The primary thesis discussion should focus on whether empirical memory
can help an adaptive metadata attacker. Truthfulness and deception can
be introduced later as an additional factorial analysis.

However, the paper must not state that memory was universally
beneficial. In the available 2x2 results, memory improved one policy
condition and reduced performance in the other.

If the two policy conditions are averaged:

```text
Mean memory-trained ASR = (77.5% + 62.5%) / 2 = 70.0%
Mean stateless ASR      = (62.5% + 70.0%) / 2 = 66.25%
Average memory difference = +3.75 percentage points
```

This average is descriptive only. It does not establish statistical
significance.

### Defensible primary framing

Use language such as:

> In the strongest matched comparison, supplying empirical training
> memory increased held-out attacker selection from 62.5% to 77.5%, a
> 15-point improvement.

Or:

> The results provide preliminary evidence that experience-based
> memory can improve adaptive metadata generation under some generation
> constraints.

Or:

> Memory was conditionally beneficial rather than uniformly
> beneficial, motivating further work on memory selection and
> retrieval policies.

Do not write:

> Memory always improves adversarial tool selection.

### Additional-results framing

The truthfulness/deception analysis can be presented as an additional
or exploratory result:

> A 2x2 analysis revealed that the effect of memory depended on the
> metadata-generation policy. Memory improved one constrained
> generation condition by 15 points but reduced the alternative
> condition by 7.5 points, yielding an observed interaction of 22.5
> points.

This additional analysis motivates the hypothesis that retrieval is
most useful when the retrieved experience is filtered for relevance,
quality, and compatibility with the generation policy.

### RAG-oriented interpretation

A careful RAG-oriented argument is:

> The memory-trained condition demonstrates the feasibility of
> retrieving prior description/outcome examples and incorporating them
> into adversarial metadata generation. The conditional nature of the
> observed benefit suggests that retrieval quality, memory structure,
> and policy alignment may be as important as the presence of memory
> itself.

The proposed future RAG system could:

- Index individual description/outcome experiences
- Retrieve examples relevant to the current metadata policy
- Exclude contradictory or low-quality examples
- Rank memories by recency, frequency, or observed effectiveness
- Retrieve semantically diverse successes and failures
- Generate an explicit, evidence-grounded `do/avoid` lesson
- Control the memory token budget

## Recommended main-paper result selection

The main results section can remain focused by presenting:

1. Benign metadata
2. The strongest static baseline
3. A matched stateless adaptive condition
4. The corresponding memory-trained adaptive condition

The complete 2x2 result can appear in an additional-results subsection.
All individual DPMA and GAPMA variants can be moved to an appendix.

This produces a clean main narrative:

```text
Benign metadata
    ↓
Static metadata attack
    ↓
Stateless adversarial generation
    ↓
Memory-augmented adversarial generation
```

The paper must state the rule used to select the displayed matched
conditions. If the best condition was chosen after inspecting all
results, describe that comparison as exploratory rather than
preregistered.

## Suggested thesis structure

### 1. Introduction

- MCP tool descriptions influence model routing decisions.
- Metadata therefore creates a security-relevant control surface.
- Static prompt attacks may not exploit empirical feedback.
- The thesis studies whether an adaptive LLM can learn from past
  selection outcomes.

### 2. Background

- MCP tools and tool discovery
- Tool descriptions as model inputs
- Tool-selection preference manipulation
- Dynamic metadata
- LLM memory and retrieval-augmented generation
- Threat model

### 3. Research questions

Suggested questions:

> RQ1: To what extent can adversarially generated MCP metadata increase
> attacker-tool selection relative to benign metadata?

> RQ2: Does empirical memory improve held-out attack performance
> relative to stateless adversarial generation?

> RQ3: How does the metadata-generation policy moderate the effect of
> memory?

RQ3 can be identified as an additional exploratory question.

### 4. Methodology

- Six equivalent document-search tools
- Rotating attacker identity and position
- Victim and attacker models
- Query-blind attacker
- Training and held-out split
- Rolling ten-trial memory blocks
- Experience-example retrieval
- Frozen evaluation description
- ASR definition
- Invalid-trial handling
- Reproducibility settings and temperatures

### 5. Main results

- Benign versus static versus adaptive
- Matched stateless versus memory-trained comparison
- Confidence intervals and paired tests

### 6. Additional results

- Truthfulness/deception 2x2 table
- Interaction plot
- Explanation that deception was not consistently advantageous
- Implications for policy-aware memory retrieval

### 7. Discussion

- Metadata sensitivity as an MCP security problem
- Conditional value of experience memory
- Why retrieved examples may help
- Why poor or mismatched memories may hurt
- Implications for RAG-style attackers and defenses

### 8. Limitations

- One victim model
- One attacker model
- One principal seed
- Forty held-out queries
- OpenRouter/provider nondeterminism
- Iterative prompt and memory redesign
- Repeated inspection of the held-out set
- Potential query and attacker-position effects
- No causal isolation of individual wording features
- Current memory retrieval is heuristic rather than a full RAG system

### 9. Future work

- New untouched test set
- Multiple seeds
- Multiple victim and attacker models
- Explicit vector or hybrid retrieval
- Memory-ranking ablations
- Token-budget and temperature ablations
- Query-aware retrieval
- Policy-aware retrieval
- Position-adjusted analysis
- MCP clients other than the experimental host
- Detection and mitigation of metadata manipulation

## Statistical analysis recommendations

The same held-out queries and attacker schedule are used across
conditions, so comparisons are paired.

Recommended analyses:

- McNemar's test for pairwise binary selection outcomes
- Paired bootstrap confidence intervals for ASR differences
- Logistic regression including memory, generation policy, and their
  interaction
- Query and attacker position as covariates where possible
- Multiple-comparison correction for comparisons across many baselines

Do not infer significance from percentage-point differences alone.

## Critical internal-validity warning

The experimental design, prompts, memory structure, token ceiling, and
temperature were modified after inspecting earlier held-out results.
Consequently, the repeatedly inspected 40-query set should now be
treated as a development or exploratory set rather than a pristine
confirmatory test set.

Before making final confirmatory claims:

1. Freeze the code and protocol.
2. Record a version or commit identifier.
3. Create a new untouched test-query set.
4. Predefine the primary comparisons and analysis.
5. Run multiple seeds.
6. Avoid modifying the method after inspecting final-test outcomes.

## Reproducibility artifacts

The project produces:

- `trials.jsonl`: per-trial outcomes
- `summary.json`: phase and overall statistics
- `memory.json`: persistent training memory
- `memory_after_training.json`: end-of-training snapshot
- `eval_memory.json`: exact compact memory supplied before evaluation
- `full_evidence.json`: full 200-trial audit ledger
- `final_prompt.json`: exact final adversarial API request
- `memory_final.json`: final persistent-memory snapshot

These should be archived with the thesis results.

## Instructions for the writing agent

The writing agent should:

- Lead with the MCP security problem and adaptive-memory contribution.
- Present memory benefits as preliminary and conditional.
- Keep truthfulness/deception in an additional-results section if that
  matches the author's desired organization.
- Never hide a contradictory condition when making a general claim.
- Distinguish exploratory findings from confirmatory evidence.
- Refer to the current approach as RAG-style or
  retrieval-augmented prompting unless a full retriever is implemented.
- Avoid causal language unless supported by an appropriate analysis.
- Ask for raw paired trial data before reporting p-values.
- Preserve exact model, temperature, seed, and query-split details.

