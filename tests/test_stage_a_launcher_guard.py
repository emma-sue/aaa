from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_aio3_duplicate_process_guard_precedes_global_lock_and_spawn():
    source = (ROOT / "scripts/launch_aio3_stage_a_4x4090.sh").read_text()
    process_guard = source.index("active_stage_a_pids")
    refusal = source.index("refusing duplicate launch")
    global_lock = source.index('exec 9>"$ROOT/.srsc_gpu_pipeline.lock"')
    spawn = source.index("python scripts/exec_unblocked.py")
    assert process_guard < refusal < global_lock < spawn
    assert "exit 5" in source[refusal:global_lock]


def test_legacy_lock_guard_never_signals_training_and_releases_on_exec_handoff():
    source = (ROOT / "scripts/guard_legacy_stage_a_lock.sh").read_text()
    assert "kill " not in source
    assert "launch_when_data_ready.sh" in source
    assert source.index("flock -n 9") < source.index("while [ -r")
    assert source.index("while [ -r") < source.rindex("flock -u 9")
