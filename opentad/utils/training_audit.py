from dataclasses import dataclass
import math

import torch


class TrainingUpdateAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingUpdateAuditReport:
    trainable_parameters: tuple
    parameter_stats: dict
    module_summaries: dict
    missing_gradient_modules: tuple
    missing_update_modules: tuple

    @property
    def passed(self):
        return not self.missing_gradient_modules and not self.missing_update_modules

    def raise_for_errors(self):
        if self.passed:
            return self
        raise TrainingUpdateAuditError(
            "training update audit failed: "
            f"missing_gradient_modules={list(self.missing_gradient_modules)}; "
            f"missing_update_modules={list(self.missing_update_modules)}"
        )

    def as_dict(self):
        return {
            "passed": self.passed,
            "trainable_parameters": list(self.trainable_parameters),
            "parameter_stats": self.parameter_stats,
            "module_summaries": self.module_summaries,
            "missing_gradient_modules": list(self.missing_gradient_modules),
            "missing_update_modules": list(self.missing_update_modules),
        }


def _unwrap_model(model):
    return getattr(model, "module", model)


def snapshot_trainable_parameters(model):
    model = _unwrap_model(model)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _matches_prefix(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def audit_training_update(
    model,
    before,
    required_module_prefixes=(),
    min_grad_norm=0.0,
    min_delta_norm=0.0,
):
    model = _unwrap_model(model)
    parameter_stats = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in before:
            raise TrainingUpdateAuditError(f"parameter snapshot is missing trainable parameter: {name}")
        gradient = parameter.grad
        grad_norm = 0.0 if gradient is None else float(gradient.detach().float().norm().item())
        delta = parameter.detach().cpu().float() - before[name].float()
        delta_norm = float(delta.norm().item())
        parameter_stats[name] = {
            "grad_norm": grad_norm,
            "delta_norm": delta_norm,
            "has_finite_grad": gradient is not None and bool(torch.isfinite(gradient).all().item()),
            "nonzero_grad": math.isfinite(grad_norm) and grad_norm > float(min_grad_norm),
            "changed": math.isfinite(delta_norm) and delta_norm > float(min_delta_norm),
        }

    module_summaries = {}
    missing_gradient_modules = []
    missing_update_modules = []
    for prefix in tuple(str(value) for value in required_module_prefixes):
        names = [name for name in parameter_stats if _matches_prefix(name, prefix)]
        summary = {
            "trainable_parameters": len(names),
            "nonzero_grad_parameters": sum(parameter_stats[name]["nonzero_grad"] for name in names),
            "changed_parameters": sum(parameter_stats[name]["changed"] for name in names),
            "grad_norm_sum": float(sum(parameter_stats[name]["grad_norm"] for name in names)),
            "delta_norm_sum": float(sum(parameter_stats[name]["delta_norm"] for name in names)),
        }
        module_summaries[prefix] = summary
        if summary["nonzero_grad_parameters"] == 0:
            missing_gradient_modules.append(prefix)
        if summary["changed_parameters"] == 0:
            missing_update_modules.append(prefix)

    return TrainingUpdateAuditReport(
        trainable_parameters=tuple(sorted(parameter_stats)),
        parameter_stats=parameter_stats,
        module_summaries=module_summaries,
        missing_gradient_modules=tuple(missing_gradient_modules),
        missing_update_modules=tuple(missing_update_modules),
    )
