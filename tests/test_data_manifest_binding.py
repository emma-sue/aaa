from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_prepare_data():
    path = PROJECT_ROOT / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("srsc_prepare_data_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_data_rejects_list_drift_from_existing_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_prepare_data()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    split = tmp_path / "artifacts/manifests/locked_split_aio5.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({
        "protocol": "aio5",
        "list_sha256": {"noisy/denoise.txt": "a" * 64},
    }))
    module.assert_locked_split_list_binding(
        "aio5", {"noisy/denoise.txt": "a" * 64}
    )
    with pytest.raises(RuntimeError, match="official training-list drift"):
        module.assert_locked_split_list_binding(
            "aio5", {"noisy/denoise.txt": "b" * 64}
        )
