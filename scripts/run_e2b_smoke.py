"""Run WikiTrace's one-task OpenAI PlanCraft smoke test in an E2B Cloud VM.

Usage: uv run --with e2b python scripts/run_e2b_smoke.py
       python scripts/run_e2b_smoke.py --dry-run
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/user/wikitrace"
CONFIG = "configs/plancraft-openai-e2b-smoke.yaml"
OUTPUTS = "outputs/openai-e2b-smoke"
EXCLUDED = {".git", ".venv", ".runtime", "data", "outputs", "tools/pi-mono"}


def project_archive() -> tuple[bytes, list[str]]:
    """Bundle the working tree, including local edits, without ignored files or secrets."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if any(part == ".env" or (part.startswith(".env.") and part != ".env.example")
               for part in relative.parts):
            continue
        source = ROOT / relative
        if source.is_file() and not source.is_symlink():
            paths.append(relative)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in paths:
            archive.add(ROOT / relative, arcname=relative.as_posix(), recursive=False)
    return buffer.getvalue(), [path.as_posix() for path in paths]


def credential(name: str) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from wikitrace.harness.llm.credentials import env_value

    return env_value(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="check upload bundle without E2B")
    parser.add_argument("--offline-only", action="store_true", help="run unit tests without OpenAI")
    args = parser.parse_args()

    bundle, paths = project_archive()
    if CONFIG not in paths:
        raise RuntimeError(f"missing {CONFIG} from E2B upload bundle")
    print(f"Bundle: {len(paths)} files, {len(bundle) / 1024 / 1024:.1f} MiB")
    if args.dry_run:
        print("Excluded .env and ignored runtime, data, and outputs directories.")
        return 0

    e2b_key = credential("E2B_API_KEY")
    openai_key = "" if args.offline_only else credential("OPENAI_API_KEY")
    if not e2b_key:
        parser.error("E2B_API_KEY is missing; add it to .env or your environment")
    if not args.offline_only and not openai_key:
        parser.error("OPENAI_API_KEY is missing; add it to .env or your environment")

    try:
        from e2b import Sandbox
    except ImportError as exc:
        raise SystemExit("Install E2B SDK: uv run --with e2b python scripts/run_e2b_smoke.py") from exc

    # The SDK reads this variable. Neither key is written into the uploaded checkout.
    os.environ["E2B_API_KEY"] = e2b_key
    sandbox = None

    def scrub(value: str) -> str:
        for secret in (e2b_key, openai_key):
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value

    def run(
        label: str,
        command: str,
        *,
        cwd: str = REMOTE_ROOT,
        envs: dict[str, str] | None = None,
    ) -> None:
        print(f"\n[{label}]", flush=True)
        result = sandbox.commands.run(
            command,
            cwd=cwd,
            envs=envs or {},
            timeout=0,
            on_stdout=lambda chunk: print(scrub(chunk), end="", flush=True),
            on_stderr=lambda chunk: print(scrub(chunk), end="", file=sys.stderr, flush=True),
        )
        if result.exit_code != 0:
            raise RuntimeError(f"{label} failed with exit code {result.exit_code}")

    try:
        sandbox = Sandbox.create(timeout=3600)
        print(f"E2B sandbox: {sandbox.sandbox_id}", flush=True)
        sandbox.files.write("/tmp/wikitrace-src.tar.gz", bundle)
        run(
            "unpack",
            f"mkdir -p {REMOTE_ROOT} && tar -xzf /tmp/wikitrace-src.tar.gz -C {REMOTE_ROOT}",
            cwd="/home/user",
        )
        run("runtime", "bash scripts/setup/e2b_runtime.sh")
        prefix = 'export PATH="$HOME/.local/bin:$PATH"; '
        run("Python environment", prefix + "uv sync --extra dev --python 3.11")
        if args.offline_only:
            run("offline unit tests", prefix + "uv run pytest -q tests/unit")
            return 0
        run("Pi agent", prefix + "bash scripts/setup/pi.sh")
        run("PlanCraft simulator", prefix + "bash scripts/setup/plancraft.sh")
        api_env = {"OPENAI_API_KEY": openai_key}
        command = prefix + "uv run wikitrace "
        common = f"--config {CONFIG} --benchmark plancraft --suite plancraft-seen --trial 0"
        run("build one training task", command + f"build {common}", envs=api_env)
        run("run one test task", command + f"run {common} --method wikitrace", envs=api_env)
        print("\nE2B WikiTrace smoke test completed.")
        return 0
    finally:
        if sandbox is not None:
            try:
                result = sandbox.commands.run(
                    f"if [ -d {OUTPUTS} ]; then tar -czf /tmp/wikitrace-e2b-results.tar.gz {OUTPUTS}; fi",
                    cwd=REMOTE_ROOT,
                    timeout=120,
                )
                if result.exit_code == 0:
                    data = sandbox.files.read("/tmp/wikitrace-e2b-results.tar.gz", format="bytes")
                    destination = ROOT / "outputs" / "wikitrace-e2b-results.tar.gz"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    print(f"Saved results: {destination}")
            except Exception as exc:
                print(f"Could not download partial results: {scrub(str(exc))}", file=sys.stderr)
            sandbox.kill()


if __name__ == "__main__":
    raise SystemExit(main())
