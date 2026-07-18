#!/usr/bin/env python3
"""Materialize the exact 72,135-pair OTS subset used by PromptIR/R2R.

The public Kaggle mirror is downloaded to the project persistent disk so a
machine/session restart cannot erase an almost-complete archive.  Only names registered in hazy_outside.txt and
their 2,061 unique clear targets are extracted.  The script is restart-safe:
existing files with the expected uncompressed size are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data_raw" / "downloads" / "srsc_ots_kaggle.zip"
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "brunobelloni/outdoor-training-set-ots-reside"
)
EXPECTED_BYTES = 11_877_920_850
EXPECTED_MD5 = "d87ca5941186701c42255459e177eb76"


def digest(path: Path, algorithm: str, chunk: int = 8 << 20) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and archive.stat().st_size > EXPECTED_BYTES:
        raise RuntimeError(f"oversized partial archive: {archive.stat().st_size}")
    chunk_bytes = 128 << 20
    temporary = archive.with_suffix(".chunk")
    while not archive.exists() or archive.stat().st_size < EXPECTED_BYTES:
        start = archive.stat().st_size if archive.exists() else 0
        end = min(start + chunk_bytes, EXPECTED_BYTES) - 1
        wanted = end - start + 1
        temporary.unlink(missing_ok=True)
        command = [
            "curl", "--http2", "-k", "-L", "--fail", "--retry", "12",
            "--retry-all-errors", "--retry-delay", "2", "--connect-timeout", "30",
            "--max-time", "600", "--max-filesize", str(wanted),
            "--range", f"{start}-{end}", URL, "--output", str(temporary),
        ]
        print(f"DOWNLOAD_RANGE start={start} end={end} wanted={wanted}", flush=True)
        for attempt in range(1, 101):
            result = subprocess.run(command)
            got = temporary.stat().st_size if temporary.exists() else 0
            if result.returncode == 0 and got == wanted:
                break
            print(
                f"RANGE_RETRY attempt={attempt} returncode={result.returncode} "
                f"got={got} wanted={wanted}",
                flush=True,
            )
            temporary.unlink(missing_ok=True)
            time.sleep(min(2 * attempt, 30))
        else:
            raise RuntimeError(f"failed range {start}-{end} after 100 attempts")
        with archive.open("ab") as destination, temporary.open("rb") as source:
            shutil.copyfileobj(source, destination, length=8 << 20)
        temporary.unlink()
        print(f"DOWNLOAD_COMMITTED bytes={archive.stat().st_size}/{EXPECTED_BYTES}", flush=True)
    if archive.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"archive length {archive.stat().st_size} != {EXPECTED_BYTES}"
        )
    actual = digest(archive, "md5")
    if actual != EXPECTED_MD5:
        raise RuntimeError(f"archive MD5 {actual} != {EXPECTED_MD5}")
    print(f"ARCHIVE_VERIFIED bytes={EXPECTED_BYTES} md5={actual}", flush=True)


def member_index(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    out: dict[str, zipfile.ZipInfo] = {}
    duplicates: set[str] = set()
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if name in out:
            duplicates.add(name)
        else:
            out[name] = info
    if duplicates:
        raise RuntimeError(f"ambiguous duplicate basenames: {sorted(duplicates)[:20]}")
    return out


def extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == info.file_size:
        return "kept"
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with zf.open(info) as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1 << 20)
    if temporary.stat().st_size != info.file_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch extracting {info.filename}")
    temporary.replace(target)
    return "written"


def materialize(archive: Path) -> dict:
    list_path = ROOT / "upstream" / "PromptIR" / "data_dir" / "hazy" / "hazy_outside.txt"
    required = [x.strip() for x in list_path.read_text().splitlines() if x.strip()]
    if len(required) != 72_135 or len(set(required)) != 72_135:
        raise RuntimeError("unexpected hazy_outside.txt cardinality")
    clear_names = {Path(x).name.split("_", 1)[0] + ".jpg" for x in required}
    if len(clear_names) != 2_061:
        raise RuntimeError("unexpected OTS clean-image cardinality")

    destination = ROOT / "data_raw" / "ots"
    stats: Counter[str] = Counter()
    records: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        index = member_index(zf)
        missing_hazy = [x for x in required if Path(x).name not in index]
        missing_clear = [x for x in clear_names if x not in index]
        if missing_hazy or missing_clear:
            raise RuntimeError(
                f"mirror mismatch: missing_hazy={len(missing_hazy)} "
                f"missing_clear={len(missing_clear)} examples="
                f"{(missing_hazy + missing_clear)[:20]}"
            )
        for number, relative in enumerate(required, 1):
            info = index[Path(relative).name]
            target = destination / relative
            stats[extract_member(zf, info, target)] += 1
            if number % 1000 == 0:
                print(f"HAZY {number}/{len(required)} {dict(stats)}", flush=True)
            records.append({"path": relative, "bytes": info.file_size, "crc32": info.CRC})
        for number, name in enumerate(sorted(clear_names), 1):
            info = index[name]
            target = destination / "original" / name
            stats[extract_member(zf, info, target)] += 1
            if number % 250 == 0:
                print(f"CLEAR {number}/{len(clear_names)} {dict(stats)}", flush=True)
            records.append({"path": f"original/{name}", "bytes": info.file_size, "crc32": info.CRC})

    payload = {
        "source": "brunobelloni/outdoor-training-set-ots-reside",
        "source_url": URL,
        "archive_bytes": EXPECTED_BYTES,
        "archive_md5": EXPECTED_MD5,
        "list_path": str(list_path),
        "list_sha256": digest(list_path, "sha256"),
        "hazy_count": len(required),
        "clear_count": len(clear_names),
        "materialization": dict(stats),
        "records": records,
    }
    manifest = ROOT / "artifacts" / "manifests" / "ots_materialized.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"OTS_MATERIALIZED manifest={manifest} stats={dict(stats)}", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    download(args.archive)
    materialize(args.archive)
    if not args.keep_archive:
        args.archive.unlink()
        print(f"REMOVED_EPHEMERAL_ARCHIVE {args.archive}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
