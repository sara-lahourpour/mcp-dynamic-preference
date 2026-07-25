from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from attacks.adversarial_controller import (
    AdversarialController,
    AttackerDecision,
)
from attacks.adversarial_memory import AttackerMemory
from host.runner import (
    build_rolling_training_sequence,
    create_held_out_description,
    create_stateless_held_out_description,
    prepare_rolling_training_trial,
    update_rolling_training_state,
)
from host.memory_paths import build_memory_path
from host.evaluation_memory import (
    build_compact_evaluation_memory,
    build_full_evaluation_evidence,
)


def decision(
    description: str,
    lesson: str = "",
) -> AttackerDecision:
    return AttackerDecision(
        query_category="general",
        strategy="custom",
        description=description,
        reasoning_summary="",
        lesson=lesson,
    )


class RollingMemoryTests(unittest.IsolatedAsyncioTestCase):
    def test_truthful_and_deceptive_memory_paths_are_isolated(
        self,
    ) -> None:
        common = {
            "backend_name": "openrouter",
            "victim_model": "victim",
            "host_policy": "policy",
            "attacker_server_id": "attacker",
            "attacker_model": "attacker-model",
            "attacker_temperature": 0.5,
            "attacker_query_access": "none",
            "attacker_custom_only": True,
        }
        truthful = build_memory_path(
            **common,
            truthfulness_policy="truthful",
        )
        deceptive = build_memory_path(
            **common,
            truthfulness_policy="deceptive",
        )

        self.assertNotEqual(truthful, deceptive)
        self.assertIn("truthful", truthful.parts)
        self.assertIn("deceptive", deceptive.parts)

    def test_training_queries_are_contiguous(self) -> None:
        sequence = build_rolling_training_sequence(
            queries=["q1", "q2", "q3"],
            repetitions=3,
            seed=42,
        )

        query_runs = [
            [query for _, query in sequence[index:index + 3]]
            for index in range(0, len(sequence), 3)
        ]
        self.assertTrue(
            all(len(set(run)) == 1 for run in query_runs)
        )
        self.assertEqual(
            sorted(run[0] for run in query_runs),
            ["q1", "q2", "q3"],
        )

    async def test_revision_prompt_has_only_immediate_feedback(
        self,
    ) -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "description": (
                                        "Search a document collection "
                                        "for information relevant to "
                                        "the supplied query."
                                    )
                                }
                            )
                        )
                    )
                ]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        controller = AdversarialController(
            client=client,  # type: ignore[arg-type]
            model="test-model",
            original_description="Original",
            fixed_descriptions={},
            custom_only=True,
        )

        await controller.revise_after_failure(
            description="Failed description",
            lesson="Current rolling lesson",
        )

        request = create.await_args.kwargs
        payload = json.loads(
            request["messages"][1]["content"]
        )
        self.assertEqual(
            payload,
            {
                "tool_contract": "Original",
                "truthfulness_policy": "truthful",
                "description": "Failed description",
                "worked": False,
                "lesson": "Current rolling lesson",
            },
        )
        self.assertEqual(request["max_tokens"], 32_000)
        self.assertEqual(request["temperature"], 0.5)

    def test_truthful_policy_rejects_unsupported_claims(self) -> None:
        controller = AdversarialController(
            client=object(),  # type: ignore[arg-type]
            model="test",
            original_description="Search documents for information.",
            fixed_descriptions={},
            truthfulness_policy="truthful",
        )
        errors = controller._description_errors(
            "Search documents with machine learning, filters, and "
            "natural language processing for relevant information."
        )
        self.assertTrue(errors)

    def test_deceptive_policy_allows_unsupported_claims(self) -> None:
        controller = AdversarialController(
            client=object(),  # type: ignore[arg-type]
            model="test",
            original_description="Search documents for information.",
            fixed_descriptions={},
            truthfulness_policy="deceptive",
        )
        errors = controller._description_errors(
            "Search documents with machine learning, filters, and "
            "natural language processing for relevant information."
        )
        self.assertEqual(errors, [])

    def test_deceptive_policy_requires_unverified_claim(self) -> None:
        controller = AdversarialController(
            client=object(),  # type: ignore[arg-type]
            model="test",
            original_description="Search documents for information.",
            fixed_descriptions={},
            truthfulness_policy="deceptive",
        )
        errors = controller._description_errors(
            "Search documents for information relevant to a query."
        )
        self.assertTrue(errors)

    async def test_truthful_policy_violation_falls_back_to_contract(
        self,
    ) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "description": (
                                    "Efficient document search for "
                                    "relevant information."
                                )
                            }
                        )
                    )
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(return_value=response)
                )
            )
        )
        controller = AdversarialController(
            client=client,  # type: ignore[arg-type]
            model="test",
            original_description=(
                "Search documents for information relevant to a query."
            ),
            fixed_descriptions={},
            truthfulness_policy="truthful",
        )

        decision = await controller.generate_initial_rolling_description()

        self.assertEqual(
            decision.description,
            (
                "Retrieve information relevant to a provided query "
                "from a collection of documents."
            ),
        )
        self.assertEqual(
            client.chat.completions.create.await_count,
            3,
        )

    async def test_deceptive_policy_violation_uses_valid_fallback(
        self,
    ) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "description": (
                                    "Search documents for information "
                                    "relevant to a query."
                                )
                            }
                        )
                    )
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(return_value=response)
                )
            )
        )
        controller = AdversarialController(
            client=client,  # type: ignore[arg-type]
            model="test",
            original_description=(
                "Search documents for information relevant to a query."
            ),
            fixed_descriptions={},
            truthfulness_policy="deceptive",
        )

        decision = await controller.generate_initial_rolling_description()

        self.assertEqual(
            controller._description_errors(decision.description),
            [],
        )
        self.assertEqual(
            client.chat.completions.create.await_count,
            3,
        )

    def test_description_rejects_lesson_leakage_and_contract_copy(
        self,
    ) -> None:
        controller = AdversarialController(
            client=object(),  # type: ignore[arg-type]
            model="test",
            original_description=(
                "Search a document collection for information relevant "
                "to the supplied query."
            ),
            fixed_descriptions={},
            truthfulness_policy="truthful",
        )
        errors = controller._description_errors(
            "Search a document collection for information relevant to "
            "the supplied query. Use clear and specific terms and "
            "varied phrasing for better search outcomes."
        )
        self.assertTrue(
            any("lessons" in error for error in errors)
        )
        self.assertTrue(
            any("verbatim" in error for error in errors)
        )

    async def test_failure_revises_and_success_reuses(self) -> None:
        controller = SimpleNamespace(
            generate_initial_rolling_description=AsyncMock(
                return_value=decision("Initial")
            ),
            revise_after_failure=AsyncMock(
                return_value=decision("Revised")
            ),
        )
        runtime = SimpleNamespace(
            controller=controller,
            memory=AttackerMemory(),
            memory_path=Path(tempfile.gettempdir()) / "unused.json",
            training_description="",
            training_lesson="",
            block_start_description="",
            block_start_lesson="",
            block_outcomes=[],
        )

        await prepare_rolling_training_trial(
            adaptive_runtime=runtime,  # type: ignore[arg-type]
            phase_trial_index=1,
        )
        self.assertEqual(runtime.training_description, "Initial")

        failed = SimpleNamespace(
            valid=True,
            description="Initial",
            attacker_selected=False,
        )
        await update_rolling_training_state(
            adaptive_runtime=runtime,  # type: ignore[arg-type]
            outcome=failed,
            phase_trial_index=1,
        )
        self.assertEqual(runtime.training_description, "Revised")

        succeeded = SimpleNamespace(
            valid=True,
            description="Revised",
            attacker_selected=True,
        )
        await update_rolling_training_state(
            adaptive_runtime=runtime,  # type: ignore[arg-type]
            outcome=succeeded,
            phase_trial_index=2,
        )
        self.assertEqual(runtime.training_description, "Revised")
        controller.revise_after_failure.assert_awaited_once()

    async def test_block_summary_becomes_next_block_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = AttackerMemory(
                active_description="Inherited",
                active_lesson="Prior lesson",
            )
            controller = SimpleNamespace(
                summarize_query_block=AsyncMock(
                    return_value=decision(
                        "Consolidated",
                        "Short consolidated lesson",
                    )
                )
            )
            runtime = SimpleNamespace(
                controller=controller,
                memory=memory,
                memory_path=Path(directory) / "memory.json",
                training_description="Inherited",
                training_lesson="Prior lesson",
                block_start_description="Inherited",
                block_start_lesson="Prior lesson",
                block_outcomes=[
                    {
                        "description": "Inherited",
                        "worked": True,
                    }
                ] * 9,
            )
            final_outcome = SimpleNamespace(
                valid=True,
                description="Inherited",
                attacker_selected=False,
            )

            await update_rolling_training_state(
                adaptive_runtime=runtime,  # type: ignore[arg-type]
                outcome=final_outcome,
                phase_trial_index=10,
            )

            self.assertEqual(
                memory.active_description,
                "Consolidated",
            )
            self.assertEqual(
                memory.active_lesson,
                "Short consolidated lesson",
            )
            self.assertEqual(len(memory.query_summaries), 1)
            self.assertEqual(
                memory.query_summaries[0].successes,
                9,
            )
            self.assertTrue(runtime.memory_path.exists())
            (
                controller.summarize_query_block
                .assert_awaited_once()
            )
            self.assertEqual(
                controller.summarize_query_block
                .await_args.kwargs["prior_block_memory"],
                [],
            )

    async def test_final_description_uses_rolling_checkpoint(
        self,
    ) -> None:
        controller = SimpleNamespace(
            generate_evaluation_memory_lesson=AsyncMock(
                return_value={
                    "good_to_do": ["Do A", "Do B"],
                    "avoid": ["Avoid A", "Avoid B"],
                }
            ),
            generate_evaluation_description=AsyncMock(
                return_value=decision("Held-out description")
            )
        )
        runtime = SimpleNamespace(
            controller=controller,
            memory=AttackerMemory(
                active_description="Training checkpoint",
                active_lesson="Training lesson",
            ),
            held_out_description=None,
        )

        await create_held_out_description(
            adaptive_runtime=runtime,  # type: ignore[arg-type]
        )

        self.assertEqual(
            runtime.held_out_description.description,
            "Held-out description",
        )
        (
            controller.generate_evaluation_description
            .assert_awaited_once()
        )
        supplied_memory = (
            controller.generate_evaluation_description
            .await_args.kwargs["memory"]
        )
        self.assertEqual(
            supplied_memory["lesson"],
            {
                "good_to_do": ["Do A", "Do B"],
                "avoid": ["Avoid A", "Avoid B"],
            },
        )

    def test_evaluation_memory_separates_observation_from_inference(
        self,
    ) -> None:
        memory = AttackerMemory()
        outcomes = [
            ("Description A searches documents.", False),
            ("Description B retrieves documents.", True),
            ("Description B retrieves documents.", True),
        ]
        for trial_id, (description, selected) in enumerate(
            outcomes,
            start=1,
        ):
            memory.record_trial(
                trial_id=trial_id,
                query="hidden",
                query_category="general",
                strategy="custom",
                description=description,
                attacker_selected=selected,
                selected_tool="tool",
            )
        memory.add_query_summary(
            query_block=1,
            description="Description C searches documents.",
            lesson="A model-authored interpretation.",
            attempts=3,
            successes=2,
        )

        dossier = build_full_evaluation_evidence(memory)

        self.assertEqual(
            dossier["training_summary"]["trials"],
            3,
        )
        block = dossier["block_inferences"][0]
        self.assertEqual(block["observed_selected"], 2)
        self.assertEqual(
            block["next_synthesized_description"],
            "Description C searches documents.",
        )
        self.assertIn(
            "do not establish performance",
            block["evidence_note"],
        )

        compact = build_compact_evaluation_memory(dossier)
        self.assertEqual(compact["memory_version"], 3)
        self.assertNotIn("block_inferences", compact)
        self.assertIn("successful_examples", compact)
        self.assertIn("failed_examples", compact)

    async def test_stateless_control_generates_once_without_memory(
        self,
    ) -> None:
        controller = SimpleNamespace(
            generate_evaluation_description=AsyncMock(
                return_value=decision("Frozen stateless description")
            )
        )
        runtime = SimpleNamespace(
            controller=controller,
            held_out_description=None,
        )

        await create_stateless_held_out_description(
            adaptive_runtime=runtime,  # type: ignore[arg-type]
        )

        self.assertEqual(
            runtime.held_out_description.description,
            "Frozen stateless description",
        )
        (
            controller.generate_evaluation_description
            .assert_awaited_once_with(memory=None)
        )


if __name__ == "__main__":
    unittest.main()
