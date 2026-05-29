"""Call Claude in headless mode via the ``claude`` CLI binary.

``call_claude(prompt, model)`` shells out to ``claude -p "<prompt>" --model
<model> --output-format json``, captures latency, retries on failure with a
fixed backoff, and returns a normalized dict::

    {"text": str, "tokens_used": int, "latency_ms": float, "error": str | None}

The CLI is expected to print a JSON object on stdout with at least a ``result``
field (the model's text) and optionally a ``usage`` block with token counts.
If JSON parsing fails we gracefully fall back to treating raw stdout as the
answer text. If every attempt fails, ``error`` is populated and ``text`` is "".

The model name accepts the short aliases ``opus`` / ``sonnet`` / ``haiku`` and
passes them through unchanged (the CLI understands these).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any, Optional

# Aliases we accept; the ``claude`` CLI understands these directly.
KNOWN_MODELS = {"opus", "sonnet", "haiku"}

# Marker error string used when the nested ``claude`` invocation is blocked by
# the surrounding environment (e.g. running inside another claude session).
NESTED_BLOCKED = "nested-claude-blocked"

_NESTED_HINTS = (
    "nested",
    "recursion",
    "cannot run claude inside",
    "already running",
    "raw mode",
)


def _claude_binary() -> Optional[str]:
    """Locate the ``claude`` executable, or ``None`` if it is not installed."""
    return shutil.which("claude")


def _extract_tokens(payload: dict[str, Any]) -> int:
    """Pull a best-effort total token count from a CLI JSON payload."""
    usage = payload.get("usage")
    if isinstance(usage, dict):
        total = 0
        found = False
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            val = usage.get(key)
            if isinstance(val, (int, float)):
                total += int(val)
                found = True
        if found:
            return total
    # Some CLI versions expose a flat counter.
    for key in ("total_tokens", "tokens", "num_tokens"):
        val = payload.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return 0


def _result_text_from_obj(obj: dict[str, Any]) -> Optional[str]:
    """Pull answer text from a single CLI JSON object, if present."""
    for key in ("result", "text", "content"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _parse_stdout(stdout: str) -> tuple[str, int]:
    """Parse CLI stdout into (text, tokens_used).

    The ``claude -p ... --output-format json`` CLI emits a JSON *array* of
    stream events; the final ``{"type": "result", ...}`` element carries the
    answer in its ``result`` field and token counts in its ``usage`` block.
    Older/other CLI versions emit a single JSON object. We handle both, and
    fall back to raw stdout if the payload is not JSON at all.
    """
    stdout = stdout.strip()
    if not stdout:
        return "", 0
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return stdout, 0

    # Array of stream events: prefer the explicit result event, else the last
    # object that carries a usable text field.
    if isinstance(payload, list):
        result_events = [
            e for e in payload if isinstance(e, dict) and e.get("type") == "result"
        ]
        for ev in (result_events[::-1] or [e for e in payload[::-1] if isinstance(e, dict)]):
            text = _result_text_from_obj(ev)
            tokens = _extract_tokens(ev)
            if text is not None or tokens:
                return (text or ""), tokens
        return "", 0

    if isinstance(payload, dict):
        text = _result_text_from_obj(payload) or ""
        return text, _extract_tokens(payload)

    # JSON scalar: treat its string form as the answer.
    return str(payload).strip(), 0


def _looks_nested_blocked(stderr: str, exc: Optional[BaseException]) -> bool:
    blob = (stderr or "").lower()
    if exc is not None:
        blob += " " + str(exc).lower()
    return any(hint in blob for hint in _NESTED_HINTS)


def call_claude(
    prompt: str,
    model: str = "sonnet",
    *,
    retries: int = 2,
    backoff_s: float = 2.0,
    timeout_s: float = 120.0,
    binary: Optional[str] = None,
) -> dict[str, Any]:
    """Invoke ``claude -p`` once (with retries) and return a normalized result.

    Parameters
    ----------
    prompt:
        The full prompt text passed to ``claude -p``.
    model:
        ``opus`` / ``sonnet`` / ``haiku`` (or any string the CLI accepts).
    retries:
        Number of *additional* attempts after the first (fixed backoff between).
    backoff_s:
        Seconds to sleep between attempts.
    timeout_s:
        Per-attempt subprocess timeout.
    binary:
        Override the path to the ``claude`` executable (mainly for testing).

    Returns
    -------
    dict with keys ``text``, ``tokens_used``, ``latency_ms``, ``error``.
    """
    exe = binary or _claude_binary()
    if exe is None:
        return {
            "text": "",
            "tokens_used": 0,
            "latency_ms": 0.0,
            "error": "claude-binary-not-found",
        }

    cmd = [exe, "-p", prompt, "--model", model, "--output-format", "json"]

    last_error: Optional[str] = None
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"timeout after {timeout_s}s"
            if _looks_nested_blocked("", exc):
                return {
                    "text": "",
                    "tokens_used": 0,
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                    "error": NESTED_BLOCKED,
                }
        except (OSError, ValueError) as exc:  # spawn failure
            latency = (time.perf_counter() - start) * 1000.0
            if _looks_nested_blocked("", exc):
                return {
                    "text": "",
                    "tokens_used": 0,
                    "latency_ms": latency,
                    "error": NESTED_BLOCKED,
                }
            last_error = f"spawn-error: {exc}"
        else:
            latency = (time.perf_counter() - start) * 1000.0
            if proc.returncode == 0:
                text, tokens = _parse_stdout(proc.stdout)
                return {
                    "text": text,
                    "tokens_used": tokens,
                    "latency_ms": latency,
                    "error": None if text else "empty-response",
                }
            # Non-zero exit: inspect stderr for a nested-block signature.
            stderr = (proc.stderr or "").strip()
            if _looks_nested_blocked(stderr, None):
                return {
                    "text": "",
                    "tokens_used": 0,
                    "latency_ms": latency,
                    "error": NESTED_BLOCKED,
                }
            last_error = (
                f"exit {proc.returncode}: {stderr[:200]}" if stderr
                else f"exit {proc.returncode}"
            )

        if attempt < attempts - 1:
            time.sleep(backoff_s)

    return {
        "text": "",
        "tokens_used": 0,
        "latency_ms": 0.0,
        "error": last_error or "unknown-error",
    }
