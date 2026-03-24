"""Infrastructure utilities for Step 2 obtain ingestion."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import pandas as pd
import requests


def utc_now_iso() -> str:
    """Return UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path, manifest_columns: List[str]) -> pd.DataFrame:
    """Load a manifest or return an empty dataframe with standard columns."""
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return pd.DataFrame(columns=manifest_columns)


def append_manifest_rows(
    manifest_path: Path,
    rows: List[Dict[str, object]],
    manifest_columns: List[str],
) -> None:
    """Append new rows to a manifest without dropping historical records."""
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    for column in manifest_columns:
        if column not in new_df.columns:
            new_df[column] = None
    new_df = new_df[manifest_columns]

    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path)
        output_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        output_df = new_df

    output_df.to_csv(manifest_path, index=False)


def latest_manifest_index(manifest_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Create a latest-by-file-path index for incremental checks."""
    latest: Dict[str, Dict[str, object]] = {}
    if manifest_df.empty:
        return latest

    for row in manifest_df.to_dict(orient="records"):
        file_path = str(row.get("file_path", "")).strip()
        if file_path:
            latest[file_path] = row
    return latest


def update_schema_inventory(
    schema_records: List[Dict[str, object]],
    schema_inventory_path: Path,
    schema_columns: List[str],
) -> None:
    """Merge schema records into schema inventory while preserving history."""
    if not schema_records:
        return

    now_utc = utc_now_iso()
    new_df = pd.DataFrame(schema_records)
    if new_df.empty:
        return

    grouped = (
        new_df.groupby(["source", "schema_hash", "columns_json"], dropna=False)
        .size()
        .reset_index(name="seen_count")
    )
    grouped["first_seen_utc"] = now_utc
    grouped["last_seen_utc"] = now_utc
    grouped = grouped[schema_columns]

    if schema_inventory_path.exists():
        existing = pd.read_csv(schema_inventory_path)
        for column in schema_columns:
            if column not in existing.columns:
                existing[column] = pd.NA
        merged = pd.concat([existing[schema_columns], grouped], ignore_index=True)
    else:
        merged = grouped

    merged = (
        merged.groupby(["source", "schema_hash", "columns_json"], dropna=False)
        .agg(
            first_seen_utc=("first_seen_utc", "min"),
            last_seen_utc=("last_seen_utc", "max"),
            seen_count=("seen_count", "sum"),
        )
        .reset_index()
    )
    merged = merged[schema_columns]
    merged.to_csv(schema_inventory_path, index=False)


def response_looks_like_csv(response: requests.Response, csv_mime_hints: tuple[str, ...]) -> bool:
    """Quickly check if HTTP response appears to be CSV."""
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if "text/html" in content_type:
        return False
    if content_type and content_type not in csv_mime_hints:
        # Fall back to content sniffing for non-standard download MIME types.
        pass

    probe = response.text.lstrip("\ufeff\n\r\t ")
    probe_head = probe[:500]
    if "<!doctype html" in probe_head.lower() or "<html" in probe_head.lower():
        return False

    first_line = probe_head.splitlines()[0] if probe_head.splitlines() else ""
    if "," not in first_line:
        return False

    csv_markers = (
        "date/time",
        "station name",
        "year",
        "month",
        "day",
        "ffmc",
        "dmc",
        "dc",
        "isi",
        "bui",
        "fwi",
    )
    marker_hits = sum(marker in probe_head.lower() for marker in csv_markers)
    return marker_hits >= 2


def download_to_file_atomic(
    url: str,
    destination: Path,
    csv_mime_hints: tuple[str, ...],
    timeout_seconds: int = 60,
) -> None:
    """Download text content atomically via temp file then move into place."""
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    if not response_looks_like_csv(response, csv_mime_hints):
        raise ValueError("Response did not look like CSV payload.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=destination.parent, encoding="utf-8", newline="") as tmp:
        tmp.write(response.text)
        temp_path = Path(tmp.name)

    temp_path.replace(destination)


def acquire_obtain_lock(lock_path: Path, logger) -> bool:
    """Acquire a single-run lock so obtain is not executed concurrently."""
    payload = {
        "pid": os.getpid(),
        "started_at_utc": utc_now_iso(),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
        return True
    except FileExistsError:
        # Auto-recover stale lock from crashed or externally-killed process.
        try:
            stale = json.loads(lock_path.read_text(encoding="utf-8"))
            stale_pid = int(stale.get("pid"))
            if stale_pid > 0:
                try:
                    os.kill(stale_pid, 0)
                except OSError:
                    lock_path.unlink(missing_ok=True)
                    logger.warning("Removed stale obtain lock from non-running pid %s.", stale_pid)
                    return acquire_obtain_lock(lock_path, logger)
        except Exception:  # noqa: BLE001
            pass

        details = ""
        try:
            stale = json.loads(lock_path.read_text(encoding="utf-8"))
            details = f" Existing lock metadata: {stale}"
        except Exception:  # noqa: BLE001
            details = ""

        logger.error(
            "Another obtain run appears active because lock file exists at %s.%s",
            lock_path,
            details,
        )
        logger.error("If no run is active, remove the stale lock file and rerun obtain.")
        return False


def release_obtain_lock(lock_path: Path, logger) -> None:
    """Release obtain run lock best-effort."""
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to remove obtain lock file %s: %s", lock_path, exc)
