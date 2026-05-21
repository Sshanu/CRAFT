from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent


class CodexCallError(RuntimeError):
    """Raised when the Codex subprocess fails or produces unparseable output."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class CodexClient:
    """One Codex CLI client per (model, prompt+schema) pair."""

    def __init__(
        self,
        *,
        model: str,
        prompt_path: str | Path,
        schema_path: str | Path,
        codex_binary: str = "codex",
        workdir: str | Path | None = None,
        reasoning_effort: str = "low",
        timeout_sec: int = 300,
    ) -> None:
        self.model = model
        self.codex_binary = codex_binary
        self.workdir = str(Path(workdir or REPO))
        self.prompt_path = Path(prompt_path)
        self.schema_path = Path(schema_path)
        self.reasoning_effort = reasoning_effort
        self.timeout_sec = int(timeout_sec)
        if not self.prompt_path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {self.prompt_path}")
        if not self.schema_path.is_file():
            raise FileNotFoundError(f"Schema not found: {self.schema_path}")
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def ensure_available(self) -> None:
        if shutil.which(self.codex_binary) is None:
            raise RuntimeError(
                f"codex binary {self.codex_binary!r} not on PATH. "
                "Install Codex CLI or pass codex_binary=<path>."
            )

    def render(self, fields: dict[str, Any]) -> str:
        # Plain `{name}` placeholders. We don't use str.format_map because the
        # prompt body contains LaTeX (`\frac{1}{2}`, `\boxed{...}`, etc.) and
        # any literal `{...}` that isn't an intended placeholder would either
        # raise or get consumed. Instead substitute each declared field via
        # `.replace`, leaving every other `{...}` intact.
        rendered = self.prompt_template
        for key, value in fields.items():
            rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
        return rendered

    def _build_command(self, output_path: Path) -> list[str]:
        return [
            self.codex_binary,
            "exec",
            "-m", self.model,
            "--config", f'model_reasoning_effort="{self.reasoning_effort}"',
            "-C", self.workdir,
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--color", "never",
            "--output-schema", str(self.schema_path),
            "--output-last-message", str(output_path),
        ]

    def call(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Render the prompt with `fields`, run codex, return the parsed JSON.

        Raises CodexCallError on subprocess failure, timeout, or invalid JSON.
        """
        prompt_text = self.render(fields)
        with tempfile.TemporaryDirectory(prefix="codex_") as tmpdir:
            output_path = Path(tmpdir) / "out.json"
            cmd = self._build_command(output_path)
            try:
                result = subprocess.run(
                    cmd,
                    input=prompt_text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                # Salvage path: codex CLI sometimes finishes its work and
                # writes the response file but doesn't exit cleanly within
                # the timeout. If the output file exists and parses as JSON,
                # use it instead of treating the whole call as failed —
                # otherwise we throw away a successful generation.
                if output_path.exists():
                    raw = output_path.read_text(encoding="utf-8").strip()
                    if raw:
                        try:
                            return json.loads(raw)
                        except json.JSONDecodeError:
                            pass
                raise CodexCallError(
                    f"codex timed out after {self.timeout_sec}s",
                    stdout=str(exc.stdout or ""),
                    stderr=str(exc.stderr or ""),
                ) from exc
            if result.returncode != 0:
                raise CodexCallError(
                    f"codex exit {result.returncode}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            if not output_path.exists():
                raise CodexCallError(
                    "codex produced no output file",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            raw = output_path.read_text(encoding="utf-8").strip()
            if not raw:
                raise CodexCallError(
                    "codex produced empty output",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CodexCallError(
                    f"codex output not valid JSON: {exc}",
                    stdout=raw,
                    stderr=result.stderr,
                ) from exc
