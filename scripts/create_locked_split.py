#!/usr/bin/env python3
"""Pre-register content-disjoint train/locked-val groups before metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260713
RATIOS = {"denoise": 0.02, "derain": 0.10, "dehaze": 0.02, "deblur": 0.10, "lowlight": 0.10}


def lines(path: Path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def rank(group: str) -> bytes:
    return hashlib.sha256(f"{SEED}:{group}".encode()).digest()


def choose(groups: set[str], ratio: float) -> list[str]:
    count = max(1, round(len(groups) * ratio))
    return sorted(sorted(groups, key=rank)[:count])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["aio3", "aio5"], required=True)
    args = parser.parse_args()
    list_root = ROOT / "upstream/PromptIR/data_dir" if args.protocol == "aio3" else Path("/root/R2R/data_dir")
    groups: dict[str, set[str]] = {}
    groups["denoise"] = {f"denoise/{Path(x).name}" for x in lines(list_root / "noisy/denoise.txt")}
    groups["derain"] = {
        f"derain/norain-{Path(x).name.split('rain-', 1)[-1].rsplit('.', 1)[0]}"
        for x in lines(list_root / "rainy/rainTrain.txt")
    }
    groups["dehaze"] = {
        f"dehaze/{Path(x).name.split('_', 1)[0]}" for x in lines(list_root / "hazy/hazy_outside.txt")
    }
    if args.protocol == "aio5":
        groups["deblur"] = {
            f"deblur/{Path(x).stem.split('-', 1)[0]}" for x in lines(list_root / "gopro/train_gopro.txt")
        }
        groups["lowlight"] = {
            f"lowlight/{Path(x).stem}" for x in lines(list_root / "lol/train_lol.txt")
        }
    selected = {task: choose(values, RATIOS[task]) for task, values in groups.items()}
    locked = sorted(group for values in selected.values() for group in values)
    hashes = {}
    for path in sorted(list_root.rglob("*.txt")):
        hashes[str(path.relative_to(list_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "protocol": args.protocol,
        "seed": SEED,
        "policy": "sha256-ranked clean-content groups; fixed before model metrics",
        "ratios": {task: RATIOS[task] for task in groups},
        "available_group_counts": {task: len(values) for task, values in groups.items()},
        "locked_group_counts": {task: len(values) for task, values in selected.items()},
        "locked_by_task": selected,
        "locked_groups": locked,
        "list_sha256": hashes,
    }
    out = ROOT / "artifacts/manifests" / f"locked_split_{args.protocol}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("protocol", "available_group_counts", "locked_group_counts")}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
