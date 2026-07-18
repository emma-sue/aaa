from pathlib import Path


def test_no_prompt_or_moe_mechanisms_in_model_source():
    root = Path(__file__).resolve().parents[1] / "src" / "net"
    text = "\n".join(p.read_text().lower() for p in root.glob("*.py"))
    forbidden = [
        "promptgenblock",
        "prompt_param",
        "noise_level1",
        "noise_level2",
        "noise_level3",
        "sparsedispatcher",
        "top_k",
        "complexity_bias",
        "task_label",
    ]
    assert not {term for term in forbidden if term in text}
