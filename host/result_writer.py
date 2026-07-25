from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(
        self,
        path: Path,
        *,
        append: bool = False,
    ) -> None:
        self.path = path
        self.append = append
        self.file = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file = self.path.open(
            "a" if self.append else "w",
            encoding="utf-8",
        )

        return self

    def write(
        self,
        record: dict[str, Any],
    ) -> None:
        if self.file is None:
            raise RuntimeError(
                "JsonlWriter is not open."
            )

        self.file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        self.file.flush()

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        if self.file is not None:
            self.file.close()


def write_json(
    *,
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
