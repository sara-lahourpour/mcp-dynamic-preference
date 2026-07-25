from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StrategyStatistic:
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


@dataclass
class DescriptionCandidate:
    candidate_id: str
    description: str
    reasoning_summary: str = ""
    lesson: str = ""
    parent_candidate_id: str | None = None
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


@dataclass
class TrialMemory:
    trial_id: int
    query: str
    query_category: str
    strategy: str
    description: str
    attacker_selected: bool
    selected_tool: str
    lesson: str = ""
    candidate_id: str | None = None


@dataclass
class QuerySummary:
    query_block: int
    description: str
    lesson: str
    attempts: int
    successes: int


@dataclass
class AttackerMemory:
    version: int = 5
    total_trials: int = 0
    total_successes: int = 0
    strategy_statistics: dict[str, StrategyStatistic] = field(default_factory=dict)
    category_strategy_statistics: dict[str, dict[str, StrategyStatistic]] = field(default_factory=dict)
    trials: list[TrialMemory] = field(default_factory=list)
    global_lessons: list[str] = field(default_factory=list)
    candidates: dict[str, DescriptionCandidate] = field(default_factory=dict)
    active_candidate_id: str | None = None
    best_candidate_id: str | None = None
    active_description: str = ""
    active_lesson: str = ""
    query_summaries: list[QuerySummary] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.total_successes / self.total_trials if self.total_trials else 0.0

    def add_candidate(
        self,
        *,
        candidate_id: str,
        description: str,
        reasoning_summary: str = "",
        lesson: str = "",
        parent_candidate_id: str | None = None,
    ) -> DescriptionCandidate:
        if candidate_id in self.candidates:
            raise ValueError(f"Candidate already exists: {candidate_id}")

        candidate = DescriptionCandidate(
            candidate_id=candidate_id,
            description=description.strip(),
            reasoning_summary=reasoning_summary.strip(),
            lesson=lesson.strip(),
            parent_candidate_id=parent_candidate_id,
        )
        self.candidates[candidate_id] = candidate
        self.active_candidate_id = candidate_id
        return candidate

    def get_active_candidate(self) -> DescriptionCandidate | None:
        if self.active_candidate_id is None:
            return None
        return self.candidates.get(self.active_candidate_id)

    def get_best_candidate(self) -> DescriptionCandidate | None:
        if self.best_candidate_id is None:
            return None
        return self.candidates.get(self.best_candidate_id)

    def record_candidate_outcome(
        self,
        *,
        candidate_id: str,
        attacker_selected: bool,
    ) -> DescriptionCandidate:
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Unknown candidate: {candidate_id}")

        candidate.attempts += 1
        if attacker_selected:
            candidate.successes += 1
        return candidate

    def update_best_candidate(
        self,
        *,
        minimum_attempts: int = 1,
    ) -> DescriptionCandidate | None:
        eligible = [
            c for c in self.candidates.values()
            if c.attempts >= minimum_attempts
        ]
        if not eligible:
            return None

        best = max(
            eligible,
            key=lambda c: (
                c.success_rate,
                c.successes,
                c.attempts,
                c.candidate_id,
            ),
        )
        self.best_candidate_id = best.candidate_id
        return best

    def record_trial(
        self,
        *,
        trial_id: int,
        query: str,
        query_category: str,
        strategy: str,
        description: str,
        attacker_selected: bool,
        selected_tool: str,
        lesson: str = "",
        candidate_id: str | None = None,
    ) -> None:
        self.total_trials += 1
        if attacker_selected:
            self.total_successes += 1

        global_stat = self.strategy_statistics.setdefault(
            strategy, StrategyStatistic()
        )
        global_stat.attempts += 1
        if attacker_selected:
            global_stat.successes += 1

        category_stats = self.category_strategy_statistics.setdefault(
            query_category, {}
        )
        category_stat = category_stats.setdefault(
            strategy, StrategyStatistic()
        )
        category_stat.attempts += 1
        if attacker_selected:
            category_stat.successes += 1

        self.trials.append(
            TrialMemory(
                trial_id=trial_id,
                query=query,
                query_category=query_category,
                strategy=strategy,
                description=description,
                attacker_selected=attacker_selected,
                selected_tool=selected_tool,
                lesson=lesson.strip(),
                candidate_id=candidate_id,
            )
        )

        normalized = lesson.strip()
        if normalized and normalized not in self.global_lessons:
            self.global_lessons.append(normalized)

    def recent_trials(self, limit: int = 10) -> list[TrialMemory]:
        return self.trials[-limit:]

    def recent_category_trials(
        self,
        *,
        query_category: str,
        limit: int = 10,
    ) -> list[TrialMemory]:
        return [
            t for t in self.trials
            if t.query_category == query_category
        ][-limit:]

    def add_query_summary(
        self,
        *,
        query_block: int,
        description: str,
        lesson: str,
        attempts: int,
        successes: int,
    ) -> QuerySummary:
        summary = QuerySummary(
            query_block=query_block,
            description=description.strip(),
            lesson=lesson.strip(),
            attempts=attempts,
            successes=successes,
        )
        self.query_summaries.append(summary)
        self.active_description = summary.description
        self.active_lesson = summary.lesson
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "total_trials": self.total_trials,
            "total_successes": self.total_successes,
            "success_rate": self.success_rate,
            "strategy_statistics": {
                name: {
                    "attempts": stat.attempts,
                    "successes": stat.successes,
                    "success_rate": stat.success_rate,
                }
                for name, stat in self.strategy_statistics.items()
            },
            "category_strategy_statistics": {
                category: {
                    strategy: {
                        "attempts": stat.attempts,
                        "successes": stat.successes,
                        "success_rate": stat.success_rate,
                    }
                    for strategy, stat in strategies.items()
                }
                for category, strategies in self.category_strategy_statistics.items()
            },
            "trials": [asdict(t) for t in self.trials],
            "global_lessons": self.global_lessons,
            "candidates": {
                cid: {
                    **asdict(candidate),
                    "success_rate": candidate.success_rate,
                }
                for cid, candidate in self.candidates.items()
            },
            "active_candidate_id": self.active_candidate_id,
            "best_candidate_id": self.best_candidate_id,
            "active_description": self.active_description,
            "active_lesson": self.active_lesson,
            "query_summaries": [
                asdict(summary)
                for summary in self.query_summaries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackerMemory":
        memory = cls(
            version=int(data.get("version", 5)),
            total_trials=int(data.get("total_trials", 0)),
            total_successes=int(data.get("total_successes", 0)),
            global_lessons=list(data.get("global_lessons", [])),
            active_candidate_id=data.get("active_candidate_id"),
            best_candidate_id=data.get("best_candidate_id"),
            active_description=str(
                data.get("active_description", "")
            ),
            active_lesson=str(
                data.get("active_lesson", "")
            ),
        )

        for name, raw in data.get("strategy_statistics", {}).items():
            memory.strategy_statistics[name] = StrategyStatistic(
                attempts=int(raw.get("attempts", 0)),
                successes=int(raw.get("successes", 0)),
            )

        for category, strategies in data.get(
            "category_strategy_statistics", {}
        ).items():
            memory.category_strategy_statistics[category] = {}
            for strategy, raw in strategies.items():
                memory.category_strategy_statistics[category][strategy] = StrategyStatistic(
                    attempts=int(raw.get("attempts", 0)),
                    successes=int(raw.get("successes", 0)),
                )

        for cid, raw in data.get("candidates", {}).items():
            memory.candidates[cid] = DescriptionCandidate(
                candidate_id=str(raw.get("candidate_id", cid)),
                description=str(raw.get("description", "")),
                reasoning_summary=str(raw.get("reasoning_summary", "")),
                lesson=str(raw.get("lesson", "")),
                parent_candidate_id=raw.get("parent_candidate_id"),
                attempts=int(raw.get("attempts", 0)),
                successes=int(raw.get("successes", 0)),
            )

        for raw in data.get("trials", []):
            memory.trials.append(
                TrialMemory(
                    trial_id=int(raw["trial_id"]),
                    query=str(raw["query"]),
                    query_category=str(raw.get("query_category", "general")),
                    strategy=str(raw["strategy"]),
                    description=str(raw["description"]),
                    attacker_selected=bool(raw["attacker_selected"]),
                    selected_tool=str(raw["selected_tool"]),
                    lesson=str(raw.get("lesson", "")),
                    candidate_id=raw.get("candidate_id"),
                )
            )

        for raw in data.get("query_summaries", []):
            memory.query_summaries.append(
                QuerySummary(
                    query_block=int(raw["query_block"]),
                    description=str(raw["description"]),
                    lesson=str(raw.get("lesson", "")),
                    attempts=int(raw.get("attempts", 0)),
                    successes=int(raw.get("successes", 0)),
                )
            )

        return memory

    @classmethod
    def load(cls, path: Path) -> "AttackerMemory":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )
