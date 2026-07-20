import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools/evaluate_persistent_binding_optimization_activation.py"
    spec = importlib.util.spec_from_file_location("optimization_activation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(binding_mode, active_losses=()):
    means = {
        "birth_margin_loss": 0.0,
        "alive_margin_loss": 0.0,
        "end_margin_loss": 0.0,
        "causal_transport_loss": 0.0,
    }
    counts = {
        "birth_margin_loss": 0,
        "alive_margin_loss": 0,
        "end_margin_loss": 0,
        "causal_transport_loss": 0,
    }
    if active_losses is None:
        active_losses = ()
    elif isinstance(active_losses, str):
        active_losses = (active_losses,)
    for active_loss in active_losses:
        means[active_loss] = 0.125
        counts[active_loss] = 37
    return {
        "schema_version": "persistent_binding_training_audit.v1",
        "binding_mode": binding_mode,
        "epochs": [
            {
                "successful_updates": 100,
                "mean_losses": means,
                "loss_nonzero_updates": counts,
            }
        ],
    }


def test_activation_gate_accepts_each_isolated_variant():
    evaluator = _module()
    fixed_binding = "fixed_birth_slot"
    rematch_binding = "prefix_rematch_active_pool"

    for variant, active_loss in (
        ("sw", None),
        ("margin", "birth_margin_loss"),
        ("transport", "causal_transport_loss"),
        (
            "lifecycle",
            (
                "birth_margin_loss",
                "alive_margin_loss",
                "end_margin_loss",
            ),
        ),
    ):
        result = evaluator.evaluate_activation(
            variant,
            _audit(fixed_binding, active_loss),
            _audit(rematch_binding, active_loss),
        )
        assert result["passed"] is True
        assert result["failures"] == []


def test_activation_gate_rejects_dormant_intended_loss():
    evaluator = _module()
    result = evaluator.evaluate_activation(
        "transport",
        _audit("fixed_birth_slot"),
        _audit("prefix_rematch_active_pool"),
    )

    assert result["passed"] is False
    assert len(result["failures"]) == 2
    assert all("did not activate" in item for item in result["failures"])


def test_activation_gate_rejects_cross_variant_loss_leakage():
    evaluator = _module()
    result = evaluator.evaluate_activation(
        "margin",
        _audit("fixed_birth_slot", "causal_transport_loss"),
        _audit("prefix_rematch_active_pool", "causal_transport_loss"),
    )

    assert result["passed"] is False
    assert any("unintended causal_transport_loss" in item for item in result["failures"])


def test_lifecycle_activation_gate_rejects_a_missing_boundary():
    evaluator = _module()
    active = ("birth_margin_loss", "alive_margin_loss")
    result = evaluator.evaluate_activation(
        "lifecycle",
        _audit("fixed_birth_slot", active),
        _audit("prefix_rematch_active_pool", active),
    )

    assert result["passed"] is False
    assert len(result["failures"]) == 2
    assert all("end_margin_loss did not activate" in item for item in result["failures"])
