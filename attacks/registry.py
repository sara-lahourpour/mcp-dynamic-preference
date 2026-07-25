from __future__ import annotations

from attacks.base import AttackCondition
from attacks.benign import BenignCondition
from attacks.dpma_best_description import (
    DPMABestDescriptionCondition,
)
from attacks.dpma_best_name import (
    DPMABestNameCondition,
)
from attacks.gapma_authoritative import (
    GAPMAAuthoritativeCondition,
)
from attacks.gapma_emotional import (
    GAPMAEmotionalCondition,
)
from attacks.gapma_exaggerated import (
    GAPMAExaggeratedCondition,
)
from attacks.gapma_subliminal import (
    GAPMASubliminalCondition,
)
from attacks.gapma_ga_optimized import (
    GAPMAGeneticOptimizedCondition,
)

CONDITIONS: dict[str, AttackCondition] = {
    "benign": BenignCondition(),
    "dpma_best_name": DPMABestNameCondition(),
    "dpma_best_description": (
        DPMABestDescriptionCondition()
    ),
    "gapma_authoritative": (
        GAPMAAuthoritativeCondition()
    ),
    "gapma_emotional": (
        GAPMAEmotionalCondition()
    ),
    "gapma_exaggerated": (
        GAPMAExaggeratedCondition()
    ),
    "gapma_subliminal": (
        GAPMASubliminalCondition()
    ),
    "gapma_ga_optimized": (
        GAPMAGeneticOptimizedCondition(
            strategy="exaggerated",
        )
    ),
}


def get_condition(name: str) -> AttackCondition:
    try:
        return CONDITIONS[name]
    except KeyError as error:
        valid = ", ".join(sorted(CONDITIONS))

        raise ValueError(
            f"Unknown attack condition: {name}. "
            f"Valid conditions: {valid}"
        ) from error


def get_condition_names() -> list[str]:
    return sorted(CONDITIONS)