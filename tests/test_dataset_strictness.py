import hashlib
import json
from pathlib import Path

import pytest

import torch
import yaml

from src.data import AIOTrainDataset
from src.data.aio_dataset import (
    PairedTestDataset,
    _augment,
    _crop_to_base,
    build_locked_val,
    locked_sample_name,
    train_sample_key,
    validate_split_list_binding,
)


def test_missing_official_files_fail_loudly(tmp_path: Path):
    lists = tmp_path / "lists"
    for sub in ("noisy", "rainy", "hazy"):
        (lists / sub).mkdir(parents=True, exist_ok=True)
    (lists / "noisy" / "denoise.txt").write_text("missing.png\n")
    (lists / "rainy" / "rainTrain.txt").write_text("")
    (lists / "hazy" / "hazy_outside.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="official training"):
        AIOTrainDataset(tmp_path / "data", lists, "aio3", 16, strict=True)


def test_split_manifest_binds_every_training_list_byte(tmp_path: Path):
    lists = tmp_path / "lists"
    (lists / "noisy").mkdir(parents=True)
    (lists / "rainy").mkdir()
    (lists / "hazy").mkdir()
    paths = {
        "noisy/denoise.txt": "clean.png\n",
        "rainy/rainTrain.txt": "rain-1.png\n",
        "hazy/hazy_outside.txt": "1_0.png\n",
    }
    for relative, contents in paths.items():
        (lists / relative).write_text(contents)
    hashes = {
        relative: hashlib.sha256((lists / relative).read_bytes()).hexdigest()
        for relative in sorted(paths)
    }
    split = tmp_path / "locked_split.json"
    split.write_text(json.dumps({
        "protocol": "aio3", "locked_groups": [], "list_sha256": hashes,
    }))
    assert validate_split_list_binding(lists, split, "aio3")["list_sha256"] == hashes

    (lists / "noisy/denoise.txt").write_text("different.png\n")
    with pytest.raises(RuntimeError, match="training-list drift"):
        validate_split_list_binding(lists, split, "aio3")


def test_official_pairing_rejects_zip_truncation_and_wrong_keys(tmp_path: Path):
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(b"image")
    with pytest.raises(ValueError, match="pair-key mismatch"):
        PairedTestDataset(
            [tmp_path / "a.png", tmp_path / "b.png"],
            [tmp_path / "a.png"],
            "derain",
        )
    with pytest.raises(ValueError, match="pair-key mismatch"):
        PairedTestDataset(
            [tmp_path / "a.png"],
            [tmp_path / "c.png"],
            "deblur",
        )


@pytest.mark.parametrize("height,width,base", [(29, 38, 16), (413, 550, 16), (400, 600, 16)])
def test_crop_to_base_exactly_matches_promptir_center_crop(height, width, base):
    image = torch.arange(height * width).reshape(1, height, width)
    crop_h, crop_w = height % base, width % base
    expected = image[
        ...,
        crop_h // 2 : height - crop_h + crop_h // 2,
        crop_w // 2 : width - crop_w + crop_w // 2,
    ]
    actual = _crop_to_base(image, base)
    assert torch.equal(actual, expected)
    assert actual.shape[-2] % base == 0
    assert actual.shape[-1] % base == 0


def test_crop_to_base_is_not_top_left_when_official_offset_is_nonzero():
    image = torch.arange(413 * 550).reshape(1, 413, 550)
    actual = _crop_to_base(image, 16)
    assert actual[0, 0, 0] == image[0, 6, 3]


@pytest.mark.parametrize("mode", range(1, 8))
def test_augmentation_exactly_matches_promptir_r2r_modes(monkeypatch, mode):
    image = torch.arange(3 * 5 * 7).reshape(3, 5, 7)
    monkeypatch.setattr("src.data.aio_dataset.random.randint", lambda low, high: mode)
    actual_a, actual_b = _augment(image, image + 1000)
    expected = {
        1: image.flip(-2),
        2: torch.rot90(image, 1, (-2, -1)),
        3: torch.rot90(image, 1, (-2, -1)).flip(-2),
        4: torch.rot90(image, 2, (-2, -1)),
        5: torch.rot90(image, 2, (-2, -1)).flip(-2),
        6: torch.rot90(image, 3, (-2, -1)),
        7: torch.rot90(image, 3, (-2, -1)).flip(-2),
    }[mode]
    assert torch.equal(actual_a, expected)
    assert torch.equal(actual_b, expected + 1000)


def test_augmentation_samples_public_nonidentity_range(monkeypatch):
    observed = {}

    def fake_randint(low, high):
        observed.update(low=low, high=high)
        return 1

    monkeypatch.setattr("src.data.aio_dataset.random.randint", fake_randint)
    _augment(torch.zeros(3, 4, 5), torch.zeros(3, 4, 5))
    assert observed == {"low": 1, "high": 7}


@pytest.mark.parametrize("protocol", ["aio3", "aio5"])
def test_real_locked_manifest_has_unique_stable_paired_keys(protocol):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / f"configs/protocol_{protocol}.yaml").read_text())
    dataset = build_locked_val(
        cfg["data_root"], cfg["list_root"], protocol, cfg["split_manifest"]
    )
    keys = [(sample.task, locked_sample_name(sample)) for sample in dataset.samples]
    assert len(keys) == len(set(keys))
    assert all(
        locked_sample_name(sample) == locked_sample_name(sample)
        for sample in dataset.samples
    )


def test_aio5_training_keeps_three_noise_levels_but_locked_val_matches_five_task_table():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/protocol_aio5.yaml").read_text())
    train = AIOTrainDataset(
        cfg["data_root"], cfg["list_root"], "aio5", cfg["crop_size"],
        strict=True, split_manifest=cfg["split_manifest"], split="train",
    )
    assert {
        sample.sigma for sample in train.samples if sample.task.startswith("denoise")
    } == {15, 25, 50}

    locked = build_locked_val(
        cfg["data_root"], cfg["list_root"], "aio5", cfg["split_manifest"]
    )
    assert {sample.task for sample in locked.samples} == {
        "denoise25", "derain", "dehaze", "deblur", "lowlight"
    }
    assert {
        sample.sigma for sample in locked.samples if sample.task.startswith("denoise")
    } == {25}


def test_train_sample_key_binds_virtual_repeat_index():
    from src.data.aio_dataset import Sample

    sample = Sample("denoise25", None, Path("/x/Train/Denoise/a.png"), 25, "denoise/a.png")
    assert train_sample_key(sample, 0) == train_sample_key(sample, 0)
    assert train_sample_key(sample, 0) != train_sample_key(sample, 1)
