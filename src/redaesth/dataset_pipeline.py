"""Dataset acquisition helpers for the Phase 2 raw-data pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download

from .config import RedAesthConfig, config


RAW_MANIFEST_VERSION = 1
CHECKSUM_CHUNK_SIZE = 1024 * 1024
IGNORED_DIRECTORY_NAMES = {".cache", "__pycache__"}
HUGGINGFACE_PROVIDER_DIR = "huggingface"

SnapshotDownloader = Callable[[str, Path, str | None, bool], Path]


@dataclass(slots=True)
class ApprovedDataset:
    """Commercially approved dataset metadata derived from research reports."""

    dataset_id: str
    license_name: str | None
    queries: list[str]
    downloads: int | None
    likes: int | None
    last_modified: str | None
    dataset_url: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ApprovedDataset":
        """Create a typed dataset record from a JSON payload."""

        return cls(
            dataset_id=str(payload["id"]),
            license_name=payload.get("license"),
            queries=[str(item) for item in payload.get("queries", [])],
            downloads=payload.get("downloads"),
            likes=payload.get("likes"),
            last_modified=payload.get("last_modified"),
            dataset_url=payload.get("dataset_url"),
        )


@dataclass(slots=True)
class DownloadedFile:
    """Checksum metadata for one downloaded file."""

    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the file record for JSON manifests."""

        return {
            "path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path) -> Any:
    """Read a UTF-8 JSON file from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Any) -> Path:
    """Write a UTF-8 JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def dataset_storage_name(dataset_id: str) -> str:
    """Convert a dataset identifier into a stable directory name."""

    normalized = dataset_id.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._")
    return slug or "dataset"


def iter_materialized_files(root: Path) -> list[Path]:
    """List downloaded files while ignoring cache metadata directories."""

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRECTORY_NAMES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def has_materialized_files(root: Path) -> bool:
    """Return whether the dataset directory already contains usable files."""

    return root.exists() and bool(iter_materialized_files(root))


def sha256_for_file(path: Path) -> str:
    """Compute the SHA-256 checksum for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHECKSUM_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_manifest(root: Path) -> list[DownloadedFile]:
    """Collect checksum records for every downloaded file in a dataset directory."""

    return [
        DownloadedFile(
            relative_path=str(path.relative_to(root)).replace("\\", "/"),
            size_bytes=path.stat().st_size,
            sha256=sha256_for_file(path),
        )
        for path in iter_materialized_files(root)
    ]


def load_approved_datasets(report_path: Path) -> tuple[dict[str, Any], list[ApprovedDataset]]:
    """Load the approved dataset report and return typed approved entries."""

    if not report_path.exists():
        raise FileNotFoundError(f"Approved dataset report not found: {report_path}")

    payload = read_json_file(report_path)
    approved_payload = payload.get("approved")
    if not isinstance(approved_payload, list):
        raise ValueError(f"Approved dataset report is malformed: {report_path}")

    datasets = [ApprovedDataset.from_payload(item) for item in approved_payload]
    return payload, datasets


def filter_datasets(
    datasets: list[ApprovedDataset],
    dataset_ids: list[str] | None,
) -> list[ApprovedDataset]:
    """Filter the approved datasets to an optional explicit subset."""

    if not dataset_ids:
        return datasets

    wanted = {dataset_id.strip() for dataset_id in dataset_ids if dataset_id.strip()}
    filtered = [dataset for dataset in datasets if dataset.dataset_id in wanted]
    if filtered:
        return filtered

    requested = ", ".join(sorted(wanted))
    raise ValueError(f"No approved datasets matched the requested dataset IDs: {requested}")


def default_snapshot_downloader(
    dataset_id: str,
    target_dir: Path,
    token: str | None,
    force_download: bool,
) -> Path:
    """Download a Hugging Face dataset snapshot into the requested directory."""

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=token or None,
        force_download=force_download,
    )
    return target_dir


def build_dataset_record(
    dataset: ApprovedDataset,
    *,
    target_dir: Path,
    status: str,
    files: list[DownloadedFile],
) -> dict[str, Any]:
    """Build one dataset entry for the raw-data manifest."""

    total_size = sum(item.size_bytes for item in files)
    return {
        "id": dataset.dataset_id,
        "license": dataset.license_name,
        "queries": dataset.queries,
        "downloads": dataset.downloads,
        "likes": dataset.likes,
        "last_modified": dataset.last_modified,
        "dataset_url": dataset.dataset_url,
        "local_dir": str(target_dir),
        "status": status,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": [item.to_dict() for item in files],
    }


def download_approved_datasets(
    *,
    config: RedAesthConfig = config,
    approved_report_path: Path | None = None,
    dataset_ids: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    downloader: SnapshotDownloader = default_snapshot_downloader,
) -> Path:
    """Download approved datasets and write a deterministic raw-data manifest."""

    report_path = approved_report_path or config.approved_datasets_report
    report_path = config.resolve_path(report_path)
    report_payload, approved_datasets = load_approved_datasets(report_path)
    selected_datasets = filter_datasets(approved_datasets, dataset_ids)
    if not selected_datasets:
        raise ValueError(f"No approved datasets were found in {report_path}")

    provider_root = config.raw_data_dir / HUGGINGFACE_PROVIDER_DIR
    provider_root.mkdir(parents=True, exist_ok=True)
    manifest_datasets: list[dict[str, Any]] = []
    token = config.hf_token or config.huggingface_hub_token

    for dataset in selected_datasets:
        target_dir = provider_root / dataset_storage_name(dataset.dataset_id)
        if dry_run:
            status = "planned"
            files: list[DownloadedFile] = []
        else:
            should_download = force or not has_materialized_files(target_dir)
            status = "downloaded" if should_download else "reused_existing"
            if should_download:
                downloader(dataset.dataset_id, target_dir, token, force)
            files = collect_file_manifest(target_dir)
            if not files:
                raise RuntimeError(
                    f"Dataset {dataset.dataset_id} did not produce any files under {target_dir}"
                )

        manifest_datasets.append(
            build_dataset_record(dataset, target_dir=target_dir, status=status, files=files)
        )

    manifest = {
        "manifest_version": RAW_MANIFEST_VERSION,
        "generated_at": utc_timestamp(),
        "source_report": str(report_path),
        "source_report_generated_at": report_payload.get("generated_at"),
        "provider": HUGGINGFACE_PROVIDER_DIR,
        "approved_count": len(selected_datasets),
        "dry_run": dry_run,
        "datasets": manifest_datasets,
    }
    return write_json_file(config.raw_data_manifest_path, manifest)
