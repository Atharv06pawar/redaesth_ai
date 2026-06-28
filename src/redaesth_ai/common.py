from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("redaesth")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT = 30
DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "RedAesth-AI/0.1 (+https://huggingface.co; repository bootstrap)",
}


class FetchError(RuntimeError):
    """Raised when a network request cannot be completed successfully."""


@dataclass(slots=True)
class FetchResult:
    """Simple wrapper around HTTP payloads for debugging and downstream use."""

    url: str
    status: int
    body: bytes


def configure_logging(level: str = "INFO") -> None:
    """Configure repository-wide logging once."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def now_utc() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def timestamp_slug() -> str:
    """Return a compact UTC timestamp usable in filenames."""

    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def ensure_directory(path: Path) -> Path:
    """Create the directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> Path:
    """Write UTF-8 text with parent directories created automatically."""

    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON using stable formatting."""

    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def append_text(path: Path, content: str) -> Path:
    """Append text to an existing file, creating it when absent."""

    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return path


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT, accept: str | None = None) -> FetchResult:
    """Fetch a remote resource and return raw bytes."""

    headers = dict(DEFAULT_HEADERS)
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = getattr(response, "status", 200)
    except Exception as exc:  # pragma: no cover - network behavior is environment-specific
        raise FetchError(f"Failed to fetch {url}: {exc}") from exc
    return FetchResult(url=url, status=status, body=body)


def fetch_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> Any:
    """Fetch and decode JSON."""

    return json.loads(fetch(url=url, timeout=timeout, accept="application/json").body)


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch and decode UTF-8 text."""

    return fetch(url=url, timeout=timeout, accept="text/plain, text/html, application/xml").body.decode(
        "utf-8",
        errors="replace",
    )


def safe_slug(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "item"


def quote_path_component(value: str) -> str:
    """Quote an identifier while preserving forward slashes."""

    return urllib.parse.quote(value, safe="/")


def first_tag_value(tags: list[str], prefix: str) -> str | None:
    """Extract the suffix from the first tag using the given prefix."""

    for tag in tags:
        if tag.startswith(prefix):
            return tag.removeprefix(prefix)
    return None


def license_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Extract a normalized license string from Hugging Face metadata."""

    card_data = metadata.get("cardData") or {}
    if isinstance(card_data, dict):
        for key in ("license", "license_name"):
            value = card_data.get(key)
            if value:
                return str(value).strip().lower()

    tags = metadata.get("tags") or []
    if isinstance(tags, list):
        tag_value = first_tag_value([str(tag) for tag in tags], "license:")
        if tag_value:
            return tag_value.strip().lower()
    return None


def parameter_count_from_metadata(metadata: dict[str, Any]) -> int | None:
    """Extract parameter count from Hugging Face safetensors metadata when present."""

    safetensors = metadata.get("safetensors") or {}
    if not isinstance(safetensors, dict):
        return None

    total = safetensors.get("total")
    if isinstance(total, int):
        return total

    parameters = safetensors.get("parameters")
    if isinstance(parameters, dict):
        counts = [value for value in parameters.values() if isinstance(value, int)]
        if counts:
            return max(counts)
    return None


def human_params(parameter_count: int | None) -> str:
    """Render parameter count in a compact human-readable format."""

    if parameter_count is None:
        return "n/a"
    if parameter_count >= 1_000_000_000:
        return f"{parameter_count / 1_000_000_000:.2f}B"
    return f"{parameter_count / 1_000_000:.1f}M"


def next_decision_number(path: Path) -> int:
    """Determine the next decision log sequence number."""

    if not path.exists():
        return 1
    content = path.read_text(encoding="utf-8")
    numbers = [int(match.group(1)) for match in re.finditer(r"## Decision (\d+):", content)]
    return (max(numbers) + 1) if numbers else 1


def append_decision(
    *,
    title: str,
    phase: str,
    decision: str,
    alternatives: str,
    justification: str,
    impact: str,
) -> None:
    """Append a formatted decision entry to the repository decision log."""

    path = PROJECT_ROOT / "DECISION_LOG.md"
    number = next_decision_number(path)
    entry = (
        f"\n## Decision {number}: {title}\n"
        f"**Date:** {now_utc().isoformat()}\n"
        f"**Phase:** {phase}\n"
        f"**Decision:** {decision}\n"
        f"**Alternatives Considered:** {alternatives}\n"
        f"**Justification:** {justification}\n"
        f"**Impact:** {impact}\n"
        "---\n"
    )
    append_text(path, entry)


def latest_matching_file(directory: Path, prefix: str, suffix: str) -> Path | None:
    """Return the latest lexicographically matching file in a directory."""

    candidates = sorted(
        path for path in directory.glob(f"{prefix}*{suffix}") if path.is_file()
    )
    return candidates[-1] if candidates else None
