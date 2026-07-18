#!/usr/bin/env python3
"""Build a symlink-only official R2R/PromptIR data layout and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LISTS_3D = ROOT / "upstream" / "PromptIR" / "data_dir"
LISTS_5D = Path("/root/R2R/data_dir")


def list_hashes(list_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(list_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(list_root.rglob("*.txt"))
    }


def assert_locked_split_list_binding(protocol: str, hashes: dict[str, str]) -> None:
    """Reject list drift after the content-disjoint split was preregistered."""
    split = ROOT / "artifacts" / "manifests" / f"locked_split_{protocol}.json"
    if not split.is_file():
        # First-time setup creates the data manifest before create_locked_split.
        return
    try:
        payload = json.loads(split.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid locked split manifest: {split}") from error
    if payload.get("protocol") != protocol:
        raise RuntimeError(f"locked split protocol mismatch: {split}")
    frozen = payload.get("list_sha256")
    if not isinstance(frozen, dict) or frozen != hashes:
        missing = sorted(set(frozen or {}) - set(hashes))
        added = sorted(set(hashes) - set(frozen or {}))
        changed = sorted(
            key for key in set(hashes).intersection(frozen or {})
            if hashes[key] != frozen[key]
        )
        raise RuntimeError(
            "official training-list drift from locked split: "
            f"protocol={protocol} missing={missing[:8]} added={added[:8]} "
            f"changed={changed[:8]}"
        )


def link(target: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == target.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.symlink_to(target)


def link_flat_sources(sources: list[tuple[Path, str]], destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for source, prefix in sources:
        if not source.exists():
            continue
        for path in source.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                out = destination / f"{prefix}{path.name}"
                if out.exists() or out.is_symlink():
                    continue
                out.symlink_to(path)


def build():
    raw = ROOT / "data_raw"
    rain100_source = raw / "rain100l_official" / "Rain100L"
    rain100_layout = raw / "rain100l_test"
    if rain100_source.exists():
        for index in range(1, 101):
            mappings = (
                (rain100_source / "rainy" / f"rain-{index:03d}.png", rain100_layout / "input" / f"{index}.png"),
                (rain100_source / f"norain-{index:03d}.png", rain100_layout / "target" / f"{index}.png"),
            )
            for source, destination in mappings:
                if source.is_file():
                    link(source, destination)
    sots_source = Path("/root/autodl-tmp/.autodl/dehaze/sots_extracted/outdoor")
    sots_layout = raw / "sots_official"
    if sots_source.exists():
        link(sots_source / "hazy", sots_layout / "input")
        link(sots_source / "gt", sots_layout / "target")
    # Existing and downloaded denoising sources are flattened by filename.
    link_flat_sources(
        [
            (Path("/root/autodl-tmp/.autodl/denoise/BSD400"), "a"),
            # PromptIR's AIO-3 list prefixes BSD400 names with ``a`` while
            # R2R's AIO-5 list addresses the exact same files without it.
            # Keep both symlink aliases so the image bytes remain identical.
            (Path("/root/autodl-tmp/.autodl/denoise/BSD400"), ""),
            (Path("/root/autodl-tmp/.autodl/promptir_official/WaterlooED"), ""),
            (Path("/root/autodl-tmp/.autodl/promptir_official/WED"), ""),
        ],
        DATA / "Train" / "Denoise",
    )
    candidates = {
        DATA / "Train" / "Derain" / "rainy": Path(
            "/root/autodl-tmp/.autodl/promptir_official/RainTrainL/RainTrainL"
        ),
        DATA / "Train" / "Dehaze": raw / "ots",
        DATA / "Train" / "Deblur" / "blur": raw / "gopro_train" / "train" / "input",
        DATA / "Train" / "Deblur" / "sharp": raw / "gopro_train" / "train" / "target",
        DATA / "Train" / "Lowlight" / "low": raw / "lol" / "our485" / "low",
        DATA / "Train" / "Lowlight" / "high": raw / "lol" / "our485" / "high",
        DATA / "Test" / "Denoise" / "cbsd68": Path("/root/autodl-tmp/.autodl/denoise/CBSD68"),
        DATA / "Test" / "Derain" / "Rain100L": rain100_layout,
        DATA / "Test" / "Dehaze": sots_layout,
        DATA / "Test" / "Deblur" / "blur": raw / "gopro_test" / "test" / "GoPro" / "input",
        DATA / "Test" / "Deblur" / "sharp": raw / "gopro_test" / "test" / "GoPro" / "target",
        DATA / "Test" / "Lowlight" / "low": raw / "lol" / "eval15" / "low",
        DATA / "Test" / "Lowlight" / "high": raw / "lol" / "eval15" / "high",
    }
    for destination, source in candidates.items():
        if source.exists():
            link(source, destination)

    # RainTrainL stores rain/rainregion/rainstreak/GT in one flat directory.
    # The official PromptIR/R2R list uses only rain-*.png and norain-*.png.
    rain_source = Path("/root/autodl-tmp/.autodl/promptir_official/RainTrainL/RainTrainL")
    if rain_source.exists():
        gt = DATA / "Train" / "Derain" / "gt"
        gt.mkdir(parents=True, exist_ok=True)
        for path in rain_source.glob("norain-*.png"):
            out = gt / path.name
            if not out.exists() and not out.is_symlink():
                out.symlink_to(path)


def official_missing(protocol: str):
    lists = LISTS_3D if protocol == "aio3" else LISTS_5D
    specs = []
    denoise = [x.strip() for x in (lists / "noisy" / "denoise.txt").read_text().splitlines() if x.strip()]
    specs += [("denoise", DATA / "Train" / "Denoise" / Path(x).name) for x in denoise]
    rain = [x.strip() for x in (lists / "rainy" / "rainTrain.txt").read_text().splitlines() if x.strip()]
    for x in rain:
        specs.append(("derain_input", DATA / "Train" / "Derain" / x))
        specs.append(("derain_gt", DATA / "Train" / "Derain" / "gt" / ("norain-" + Path(x).name.split("rain-", 1)[-1])))
    haze = [x.strip() for x in (lists / "hazy" / "hazy_outside.txt").read_text().splitlines() if x.strip()]
    for x in haze:
        specs.append(("dehaze_input", DATA / "Train" / "Dehaze" / x))
        specs.append(("dehaze_gt", DATA / "Train" / "Dehaze" / "original" / (Path(x).name.split("_", 1)[0] + Path(x).suffix)))
    if protocol == "aio5":
        for x in [x.strip() for x in (lists / "gopro" / "train_gopro.txt").read_text().splitlines() if x.strip()]:
            specs += [("deblur_input", DATA / "Train" / "Deblur" / "blur" / x), ("deblur_gt", DATA / "Train" / "Deblur" / "sharp" / x)]
        for x in [x.strip() for x in (lists / "lol" / "train_lol.txt").read_text().splitlines() if x.strip()]:
            specs += [("lowlight_input", DATA / "Train" / "Lowlight" / "low" / x), ("lowlight_gt", DATA / "Train" / "Lowlight" / "high" / x)]
    missing = [(kind, str(path)) for kind, path in specs if not path.is_file()]
    counts = {}
    for kind, _ in specs:
        counts[kind] = counts.get(kind, 0) + 1
    return specs, missing, counts


def manifest(protocol: str):
    specs, missing, counts = official_missing(protocol)
    lists = LISTS_3D if protocol == "aio3" else LISTS_5D
    hashes = list_hashes(lists)
    assert_locked_split_list_binding(protocol, hashes)
    payload = {
        "protocol": protocol,
        "data_root": str(DATA),
        "expected_entries": len(specs),
        "missing_entries": len(missing),
        "counts": counts,
        "missing_preview": missing[:100],
        "list_sha256": hashes,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    out = ROOT / "artifacts" / "manifests" / f"{protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(encoded + "\n")
    os.replace(temporary, out)
    print(encoded)
    return 0 if not missing else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["aio3", "aio5"], required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    raise SystemExit(manifest(args.protocol))
