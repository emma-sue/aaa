import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/capture_live_stage_a_runtime.py"
    spec = importlib.util.spec_from_file_location("capture_live_stage_a_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_owners(module):
    owners = []
    for rank in range(4):
        tokens = ["python", "scripts/train_stage_a_ddp.py"]
        for flag, value in module.EXPECTED_ARGUMENTS.items():
            tokens.extend((flag, value))
        owners.append(
            {
                "pid": 1000 + rank,
                "gpu_uuid": f"GPU-{rank}",
                "cmdline": tokens,
                "environment": {
                    **module.EXPECTED_ENVIRONMENT,
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                },
            }
        )
    return owners


def test_owner_contract_accepts_exact_four_rank_runtime():
    module = load_module()
    module.validate_owners(valid_owners(module), module.DEFAULT_RUN)


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        ("missing_gpu", "exactly four"),
        ("duplicate_uuid", "UUIDs are not unique"),
        ("workers", "trainer argument mismatch"),
        ("rank", "rank coverage mismatch"),
    ],
)
def test_owner_contract_fails_closed(mutation, pattern):
    module = load_module()
    owners = valid_owners(module)
    if mutation == "missing_gpu":
        owners.pop()
    elif mutation == "duplicate_uuid":
        owners[-1]["gpu_uuid"] = owners[0]["gpu_uuid"]
    elif mutation == "workers":
        tokens = owners[-1]["cmdline"]
        tokens[tokens.index("--workers-per-rank") + 1] = "7"
    elif mutation == "rank":
        owners[-1]["environment"]["RANK"] = "2"
    with pytest.raises(RuntimeError, match=pattern):
        module.validate_owners(owners, module.DEFAULT_RUN)


def test_environment_reader_whitelists_keys(tmp_path):
    module = load_module()
    process = tmp_path / "123"
    process.mkdir()
    (process / "environ").write_bytes(
        b"WORLD_SIZE=4\0RANK=0\0LOCAL_RANK=0\0SECRET_TOKEN=must-not-leak\0"
    )
    environment = module.read_environment(tmp_path, 123)
    assert environment == {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"}
    assert "SECRET" not in json.dumps(environment)


def test_atomic_json_replaces_complete_document(tmp_path):
    module = load_module()
    output = tmp_path / "attestation.json"
    module.atomic_json(output, {"schema": "v1", "status": "PASS"})
    assert json.loads(output.read_text()) == {"schema": "v1", "status": "PASS"}
    assert not list(tmp_path.glob("*.tmp.*"))
