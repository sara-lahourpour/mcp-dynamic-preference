from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from host import result_paths


class ResultPathTests(unittest.TestCase):
    def test_unique_run_path_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(result_paths, "RESULTS_ROOT", root):
                created = result_paths.build_unique_run_paths(
                    condition="adversarial_memory",
                    run_id="test-run",
                )
                self.assertTrue(created.directory.is_dir())
                with self.assertRaises(FileExistsError):
                    result_paths.build_unique_run_paths(
                        condition="adversarial_memory",
                        run_id="test-run",
                    )

    def test_generated_run_ids_are_distinct(self) -> None:
        first = result_paths.create_run_id(
            mode="memory",
            truthfulness_policy="truthful",
            seed=42,
        )
        second = result_paths.create_run_id(
            mode="memory",
            truthfulness_policy="truthful",
            seed=42,
        )
        self.assertNotEqual(first, second)
        self.assertIn("memory_truthful_s42", first)


if __name__ == "__main__":
    unittest.main()
