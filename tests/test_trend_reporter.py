import pytest

from scripts.report_stage_a_trend import (
    enrich_aio3_aggregates,
    render_stage_a_report,
    report_already_logged,
)


def test_narrative_label_mention_does_not_suppress_real_report():
    text = "隔离测试验证 STAGE_A_TREND_EPOCH_010 不会重复。"
    assert not report_already_logged(text, "STAGE_A_TREND_EPOCH_010")


def test_dedicated_marker_is_idempotent():
    text = "<!-- STAGE_A_TREND_EPOCH_010 -->\n"
    assert report_already_logged(text, "STAGE_A_TREND_EPOCH_010")


def test_legacy_epoch5_heading_remains_idempotent():
    text = "## 2026-07-13 — STAGE_A_TREND_EPOCH_005\n"
    assert report_already_logged(text, "STAGE_A_TREND_EPOCH_005")


def test_aio3_report_names_five_setting_mean_and_computes_three_task_macro():
    row = enrich_aio3_aggregates({
        "dehaze": 30.0,
        "derain": 33.0,
        "denoise15": 36.0,
        "denoise25": 33.0,
        "denoise50": 30.0,
        "macro_psnr": 32.4,
        "epoch": 5,
        "step": 10,
    })
    assert row["five_setting_mean_psnr"] == pytest.approx(32.4)
    assert row["denoise_task_mean_psnr"] == pytest.approx(33.0)
    assert row["three_task_macro_psnr"] == pytest.approx(32.0)
    assert row["macro_psnr_semantics"] == "legacy_alias_of_five_setting_mean_psnr"


def test_aio3_report_rejects_mislabeled_legacy_macro():
    with pytest.raises(ValueError, match="not the expected five-setting mean"):
        enrich_aio3_aggregates({
            "dehaze": 30.0,
            "derain": 30.0,
            "denoise15": 30.0,
            "denoise25": 30.0,
            "denoise50": 30.0,
            "macro_psnr": 31.0,
        })


def test_render_stage_a_report_distinguishes_locked_val_from_official_claims(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("scripts.report_stage_a_trend.ROOT", tmp_path)
    rows = [
        enrich_aio3_aggregates({
            "dehaze": 30.0, "denoise15": 31.0, "denoise25": 30.0,
            "denoise50": 29.0, "derain": 32.0, "macro_psnr": 30.4,
            "epoch": 5, "step": 100,
        }),
        enrich_aio3_aggregates({
            "dehaze": 31.0, "denoise15": 32.0, "denoise25": 31.0,
            "denoise50": 30.0, "derain": 33.0, "macro_psnr": 31.4,
            "epoch": 10, "step": 200,
        }),
    ]
    report = render_stage_a_report(rows, rows[-1])
    assert "Status: **IN_PROGRESS**" in report
    assert "epoch `10`, step `200`" in report
    assert "not an official-test result" in report
    assert "31.4000000000 dB" in report
