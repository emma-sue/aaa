"""R2R/PromptIR-compatible AIO-3 and AIO-5 datasets.

The loader is strict: every official list entry must exist before training.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
EXPECTED_OFFICIAL_COUNTS = {
    "aio3": {
        "denoise15": 68,
        "denoise25": 68,
        "denoise50": 68,
        "derain": 100,
        "dehaze": 500,
    },
    "aio5": {
        "denoise25": 68,
        "derain": 100,
        "dehaze": 500,
        "deblur": 1111,
        "lowlight": 15,
    },
}


def _read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))


def _crop_to_base(image: torch.Tensor, base: int = 16) -> torch.Tensor:
    """Match PromptIR/R2R ``crop_img`` (center crop to a base multiple)."""
    h, w = image.shape[-2:]
    crop_h, crop_w = h % base, w % base
    h2, w2 = h - crop_h, w - crop_w
    if h2 == 0 or w2 == 0:
        raise ValueError(f"image {(h, w)} is smaller than evaluation base={base}")
    top, left = crop_h // 2, crop_w // 2
    return image[..., top : top + h2, left : left + w2]


def _paired_crop(a: torch.Tensor, b: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = a.shape[-2:]
    if b.shape[-2:] != (h, w):
        raise ValueError(f"pair shape mismatch: {a.shape} vs {b.shape}")
    if min(h, w) < size:
        raise ValueError(f"image {(h,w)} is smaller than crop={size}")
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return (
        a[:, top : top + size, left : left + size],
        b[:, top : top + size, left : left + size],
    )


def _augment(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match PromptIR/R2R ``random_augmentation`` exactly.

    Their public implementation deliberately samples integers 1..7, excluding
    identity mode 0.  Keep that slightly unusual distribution instead of the
    more common eight-way independent flip/transpose augmentation.
    """
    mode = random.randint(1, 7)

    def transform(image: torch.Tensor) -> torch.Tensor:
        if mode == 1:
            return image.flip(-2)
        if mode == 2:
            return torch.rot90(image, 1, (-2, -1))
        if mode == 3:
            return torch.rot90(image, 1, (-2, -1)).flip(-2)
        if mode == 4:
            return torch.rot90(image, 2, (-2, -1))
        if mode == 5:
            return torch.rot90(image, 2, (-2, -1)).flip(-2)
        if mode == 6:
            return torch.rot90(image, 3, (-2, -1))
        if mode == 7:
            return torch.rot90(image, 3, (-2, -1)).flip(-2)
        raise AssertionError(f"unreachable PromptIR/R2R augmentation mode: {mode}")

    return transform(a), transform(b)


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _current_list_sha256(list_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(list_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(list_root.rglob("*.txt"))
    }


def validate_split_list_binding(
    list_root: str | Path, split_manifest: str | Path, protocol: str,
) -> dict:
    """Bind every dataset construction to the preregistered official lists."""
    manifest_path = Path(split_manifest)
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid split manifest JSON: {manifest_path}") from error
    if payload.get("protocol") != protocol:
        raise ValueError("split manifest protocol mismatch")
    frozen = payload.get("list_sha256")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("split manifest has no frozen training-list hashes")
    actual = _current_list_sha256(Path(list_root))
    if actual != frozen:
        missing = sorted(set(frozen) - set(actual))
        added = sorted(set(actual) - set(frozen))
        changed = sorted(
            key for key in set(actual).intersection(frozen)
            if actual[key] != frozen[key]
        )
        raise RuntimeError(
            "training-list drift from split manifest: "
            f"missing={missing[:8]} added={added[:8]} changed={changed[:8]}"
        )
    return payload


@dataclass(frozen=True)
class Sample:
    task: str
    degraded: Path | None
    clean: Path
    sigma: int = 0
    group: str = ""


class AIOTrainDataset(Dataset):
    TASK_ID = {
        "denoise15": 0,
        "denoise25": 1,
        "denoise50": 2,
        "derain": 3,
        "dehaze": 4,
        "deblur": 5,
        "lowlight": 6,
    }

    def __init__(
        self,
        data_root: str | Path,
        list_root: str | Path,
        protocol: str = "aio3",
        patch_size: int = 128,
        strict: bool = True,
        split_manifest: str | Path | None = None,
        split: str = "train",
    ):
        self.data_root = Path(data_root)
        self.list_root = Path(list_root)
        self.protocol = protocol
        self.patch_size = patch_size
        if split not in {"train", "locked_val", "all"}:
            raise ValueError("split must be train, locked_val, or all")
        self.split = split
        if protocol not in {"aio3", "aio5"}:
            raise ValueError("protocol must be aio3 or aio5")
        split_payload = None
        if split_manifest is not None and split != "all":
            split_payload = validate_split_list_binding(
                self.list_root, split_manifest, protocol
            )
        self.samples = self._build_samples()
        if split_payload is not None:
            locked = set(split_payload["locked_groups"])
            self.samples = [s for s in self.samples if (s.group in locked) == (split == "locked_val")]
        missing = [s for s in self.samples if not s.clean.is_file() or (s.degraded is not None and not s.degraded.is_file())]
        if strict and missing:
            preview = "\n".join(str(s) for s in missing[:20])
            raise FileNotFoundError(f"{len(missing)} official training pairs are missing:\n{preview}")
        if not strict:
            self.samples = [s for s in self.samples if s.clean.is_file() and (s.degraded is None or s.degraded.is_file())]
        if not self.samples:
            raise RuntimeError("training dataset is empty")

    def _build_samples(self) -> list[Sample]:
        train = self.data_root / "Train"
        samples: list[Sample] = []
        denoise_names = _lines(self.list_root / "noisy" / "denoise.txt")
        denoise = [train / "Denoise" / Path(name).name for name in denoise_names]
        for sigma, task in ((15, "denoise15"), (25, "denoise25"), (50, "denoise50")):
            samples.extend(Sample(task, None, p, sigma, f"denoise/{p.name}") for p in denoise for _ in range(3))

        rain_names = _lines(self.list_root / "rainy" / "rainTrain.txt")
        rain_pairs = []
        for name in rain_names:
            degraded = train / "Derain" / name
            clean_name = "norain-" + Path(name).name.split("rain-", 1)[-1]
            clean = train / "Derain" / "gt" / clean_name
            rain_pairs.append((degraded, clean))
        rain_repeat = 120 if self.protocol == "aio3" else 80
        samples.extend(Sample("derain", d, c, group=f"derain/{c.stem}") for d, c in rain_pairs for _ in range(rain_repeat))

        haze_names = _lines(self.list_root / "hazy" / "hazy_outside.txt")
        for name in haze_names:
            degraded = train / "Dehaze" / name
            stem = Path(name).name.split("_", 1)[0]
            clean = train / "Dehaze" / "original" / f"{stem}{Path(name).suffix}"
            samples.append(Sample("dehaze", degraded, clean, group=f"dehaze/{stem}"))

        if self.protocol == "aio5":
            blur_names = _lines(self.list_root / "gopro" / "train_gopro.txt")
            samples.extend(
                Sample(
                    "deblur", train / "Deblur" / "blur" / name,
                    train / "Deblur" / "sharp" / name,
                    group=f"deblur/{Path(name).stem.split('-', 1)[0]}",
                )
                for name in blur_names
                for _ in range(30)
            )
            low_names = _lines(self.list_root / "lol" / "train_lol.txt")
            samples.extend(
                Sample(
                    "lowlight", train / "Lowlight" / "low" / name,
                    train / "Lowlight" / "high" / name,
                    group=f"lowlight/{Path(name).stem}",
                )
                for name in low_names
                for _ in range(60)
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        clean = _crop_to_base(_read_rgb(sample.clean))
        if sample.degraded is None:
            clean_u8 = (clean.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            noisy_u8 = np.clip(
                clean_u8.astype(np.float32) + np.random.randn(*clean_u8.shape) * sample.sigma,
                0,
                255,
            ).astype(np.uint8)
            degraded = torch.from_numpy(noisy_u8).permute(2, 0, 1).float() / 255.0
        else:
            degraded = _crop_to_base(_read_rgb(sample.degraded))
        degraded, clean = _paired_crop(degraded, clean, self.patch_size)
        degraded, clean = _augment(degraded, clean)
        return {
            "degraded": degraded,
            "clean": clean,
            "task": self.TASK_ID[sample.task],
            "name": sample.clean.stem,
            "sample_index": index,
            "sample_key": train_sample_key(sample, index),
        }


class PairedTestDataset(Dataset):
    def __init__(
        self,
        degraded: Iterable[Path],
        clean: Iterable[Path],
        task: str,
        *,
        degraded_key=None,
        clean_key=None,
    ):
        degraded_paths = list(degraded)
        clean_paths = list(clean)
        degraded_key = degraded_key or (lambda path: path.stem)
        clean_key = clean_key or (lambda path: path.stem)

        def keyed(paths: list[Path], key_fn, side: str) -> dict[str, Path]:
            result = {}
            for path in paths:
                key = str(key_fn(path))
                if not key or key in result:
                    raise ValueError(
                        f"duplicate/empty {task} {side} pairing key: {key!r}"
                    )
                result[key] = path
            return result

        degraded_by_key = keyed(degraded_paths, degraded_key, "degraded")
        clean_by_key = keyed(clean_paths, clean_key, "clean")
        if set(degraded_by_key) != set(clean_by_key):
            raise ValueError(
                f"{task} official pair-key mismatch: "
                f"degraded_only={sorted(set(degraded_by_key) - set(clean_by_key))[:8]} "
                f"clean_only={sorted(set(clean_by_key) - set(degraded_by_key))[:8]}"
            )
        self.pairs = [
            (degraded_by_key[key], clean_by_key[key])
            for key in sorted(degraded_by_key)
        ]
        self.task = task
        if not self.pairs or any(not a.is_file() or not b.is_file() for a, b in self.pairs):
            raise FileNotFoundError(f"invalid or empty test set: {task}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        degraded, clean = self.pairs[index]
        degraded_tensor, clean_tensor = _crop_to_base(_read_rgb(degraded)), _crop_to_base(_read_rgb(clean))
        if degraded_tensor.shape != clean_tensor.shape:
            raise ValueError(f"test pair shape mismatch: {degraded} vs {clean}")
        return {"degraded": degraded_tensor, "clean": clean_tensor, "name": degraded.stem, "task": self.task}


class DenoiseTestDataset(Dataset):
    def __init__(self, clean_paths: Iterable[Path], sigma: int, rng: np.random.RandomState | None = None):
        self.clean_paths = list(clean_paths)
        self.sigma = sigma
        self.rng = rng if rng is not None else np.random.RandomState(0)
        if not self.clean_paths:
            raise FileNotFoundError("empty denoise test set")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, index):
        clean = _crop_to_base(_read_rgb(self.clean_paths[index]))
        clean_u8 = (clean.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        noisy_u8 = np.clip(
            clean_u8.astype(np.float32) + self.rng.randn(*clean_u8.shape) * self.sigma,
            0,
            255,
        ).astype(np.uint8)
        degraded = torch.from_numpy(noisy_u8).permute(2, 0, 1).float() / 255.0
        return {"degraded": degraded, "clean": clean, "name": self.clean_paths[index].stem, "task": f"denoise{self.sigma}"}


class LockedValidationDataset(Dataset):
    """Content-disjoint, fixed full-image validation examples."""

    def __init__(self, samples: list[Sample], seed: int = 1415926):
        self.samples = samples
        self.seed = seed
        if not samples:
            raise RuntimeError("locked validation dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        clean = _crop_to_base(_read_rgb(sample.clean))
        if sample.degraded is None:
            clean_u8 = (clean.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            token = f"{self.seed}:{sample.group}:{sample.sigma}".encode()
            noise_seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
            rng = np.random.RandomState(noise_seed)
            noisy = np.clip(clean_u8.astype(np.float32) + rng.randn(*clean_u8.shape) * sample.sigma, 0, 255).astype(np.uint8)
            degraded = torch.from_numpy(noisy).permute(2, 0, 1).float() / 255.0
        else:
            degraded = _crop_to_base(_read_rgb(sample.degraded))
        if degraded.shape != clean.shape:
            raise ValueError(f"locked pair shape mismatch: {sample.degraded} vs {sample.clean}")
        return {
            "degraded": degraded,
            "clean": clean,
            "name": locked_sample_name(sample),
            "task": sample.task,
        }


def _protocol_relative_path(path: Path | None) -> str:
    """Return a machine-independent path token for paired metric identity."""
    if path is None:
        return "<synthetic-degradation>"
    parts = path.as_posix().split("/")
    for anchor in ("Train", "Test"):
        if anchor in parts:
            return "/".join(parts[parts.index(anchor):])
    return path.name


def locked_sample_name(sample: Sample) -> str:
    """Stable unique key for every selected locked-validation observation."""
    identity = "\0".join((
        sample.task,
        sample.group,
        _protocol_relative_path(sample.degraded),
        _protocol_relative_path(sample.clean),
        str(sample.sigma),
    ))
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    readable = sample.group.replace("/", "_").replace(":", "_")
    return f"{readable}:{sample.sigma}:{token}"


def train_sample_key(sample: Sample, index: int) -> str:
    """Stable identity for replay/audit without controlling augmentation RNG."""
    identity = "\0".join((
        str(index),
        sample.task,
        sample.group,
        _protocol_relative_path(sample.degraded),
        _protocol_relative_path(sample.clean),
        str(sample.sigma),
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def build_test_sets(data_root: str | Path, protocol: str) -> dict[str, Dataset]:
    root = Path(data_root) / "Test"
    clean = _images(root / "Denoise" / "cbsd68")
    sets: dict[str, Dataset] = {}
    # Official PromptIR calls np.random.seed(0) once and evaluates the three
    # sigma settings sequentially on one dataset object.  Sharing this RNG
    # reproduces those exact noise realizations.
    denoise_rng = np.random.RandomState(0)
    sigmas = (15, 25, 50) if protocol == "aio3" else (25,)
    for sigma in sigmas:
        sets[f"denoise{sigma}"] = DenoiseTestDataset(clean, sigma, denoise_rng)
    rain_root = root / "Derain" / "Rain100L"
    rain_in, rain_gt = _images(rain_root / "input"), _images(rain_root / "target")
    sets["derain"] = PairedTestDataset(rain_in, rain_gt, "derain")
    haze_root = root / "Dehaze"
    haze_in = _images(haze_root / "input")
    haze_gt_map = {p.stem: p for p in _images(haze_root / "target")}
    haze_gt = [haze_gt_map[p.stem.split("_", 1)[0]] for p in haze_in]
    sets["dehaze"] = PairedTestDataset(
        haze_in,
        haze_gt,
        "dehaze",
        degraded_key=lambda path: path.stem.split("_", 1)[0],
        clean_key=lambda path: path.stem,
    )
    if protocol == "aio5":
        blur = root / "Deblur"
        sets["deblur"] = PairedTestDataset(_images(blur / "blur"), _images(blur / "sharp"), "deblur")
        lol = root / "Lowlight"
        sets["lowlight"] = PairedTestDataset(_images(lol / "low"), _images(lol / "high"), "lowlight")
    counts = {task: len(dataset) for task, dataset in sets.items()}
    expected = EXPECTED_OFFICIAL_COUNTS[protocol]
    if counts != expected:
        raise RuntimeError(
            f"official benchmark count mismatch: protocol={protocol} "
            f"actual={counts} expected={expected}"
        )
    return sets


def build_locked_val(
    data_root: str | Path,
    list_root: str | Path,
    protocol: str,
    split_manifest: str | Path,
) -> LockedValidationDataset:
    """Build the preregistered validation set without training repeats.

    Dehaze uses five fixed atmosphere/scattering variants per held-out clean
    image.  GoPro uses every tenth frame from each held-out sequence.  AIO-5
    trains denoising on sigma 15/25/50 exactly like public R2R, but its standard
    five-task table evaluates only sigma 25, so the locked validation follows
    that same task definition.  These reductions are fixed before metrics and
    only reduce validation cost; all held-out content remains excluded from
    training.
    """
    base = AIOTrainDataset(
        data_root, list_root, protocol, patch_size=128, strict=True,
        split_manifest=split_manifest, split="locked_val",
    )
    by_group: dict[str, list[Sample]] = {}
    for sample in base.samples:
        if (
            protocol == "aio5"
            and sample.task.startswith("denoise")
            and sample.sigma != 25
        ):
            continue
        by_group.setdefault(sample.group, []).append(sample)
    selected: list[Sample] = []
    for group, samples in sorted(by_group.items()):
        unique: dict[tuple[str, str, int], Sample] = {}
        for sample in samples:
            key = (
                str(sample.degraded) if sample.degraded is not None else "",
                str(sample.clean),
                sample.sigma,
            )
            unique[key] = sample
        values = sorted(unique.values(), key=lambda x: (str(x.degraded), x.sigma))
        if group.startswith("dehaze/") and len(values) > 5:
            positions = [0, len(values) // 4, len(values) // 2, 3 * len(values) // 4, len(values) - 1]
            values = [values[position] for position in positions]
        elif group.startswith("deblur/") and len(values) > 10:
            values = values[::10]
        selected.extend(values)
    return LockedValidationDataset(selected)
