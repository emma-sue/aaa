from __future__ import annotations

import pytest

from scripts.verify_stage_a_checkpoint import expected_r2r_lr, learning_rate_integrity


def _payload(epoch: int, lr: float) -> dict:
    return {
        "epoch": epoch,
        "config": {
            "epochs": 240,
            "lr": 2e-4,
            "warmup_epochs": 15,
            "warmup_start_lr": 1e-7,
            "scheduler_max_epochs": 270,
            "pretrain_eta_min": 0.0,
        },
        "optimizer": {"param_groups": [{"lr": lr}]},
        "scheduler": {"last_epoch": epoch, "_last_lr": [lr]},
    }


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 1e-7),
        (1, 1.437857142857143e-5),
        (5, 7.149285714285714e-5),
        (10, 1.4288571428571427e-4),
        (14, 2e-4),
        (15, 2e-4),
    ],
)
def test_expected_r2r_lr_known_boundaries(epoch: int, expected: float):
    assert expected_r2r_lr(_payload(0, 1e-7)["config"], epoch) == pytest.approx(
        expected, rel=1e-14, abs=1e-16
    )


def test_learning_rate_integrity_accepts_consistent_state():
    lr = 7.149285714285714e-5
    checks, details = learning_rate_integrity(_payload(5, lr))
    assert all(checks.values())
    assert details["expected_r2r_lr"] == pytest.approx(lr, rel=1e-14)


def test_learning_rate_integrity_rejects_scheduler_epoch_or_lr_drift():
    payload = _payload(5, 7.149285714285714e-5)
    payload["scheduler"]["last_epoch"] = 4
    payload["optimizer"]["param_groups"][0]["lr"] = 1e-4
    checks, _ = learning_rate_integrity(payload)
    assert not checks["scheduler_epoch_matches_checkpoint"]
    assert not checks["optimizer_lr_matches_scheduler"]
    assert not checks["optimizer_lr_matches_r2r_closed_form"]
