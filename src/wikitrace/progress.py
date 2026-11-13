"""Concise, secret-free progress messages for interactive and CI runs."""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def progress(message: str) -> None:
    """Write a timestamped status line immediately, including in GitHub Actions."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[wikitrace {stamp}] {message}", file=sys.stderr, flush=True)
