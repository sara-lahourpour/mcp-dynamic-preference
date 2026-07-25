from __future__ import annotations

from backends.base import ModelBackend
from backends.ollama_backend import OllamaBackend


def create_backend(name: str) -> ModelBackend:
    if name == "ollama":
        return OllamaBackend()

    if name == "openrouter":
        from backends.openrouter_backend import (
            OpenRouterBackend,
        )

        return OpenRouterBackend()

    raise ValueError(f"Unknown backend: {name}")