"""Read API keys from the process environment or the repository's .env file."""

from __future__ import annotations

import os
from pathlib import Path


def env_value(name: str) -> str:
    """Environment variables win; .env is a fallback for local command-line runs."""
    if value := os.environ.get(name, "").strip():
        return value
    paths = (Path.cwd() / ".env", Path(__file__).resolve().parents[4] / ".env")
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                return value
    return ""
