"""Pre-flight optimizer coverage audits for trainable parameter plans."""

from dataclasses import dataclass


class OptimizerAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizerAuditReport:
    trainable_parameters: tuple
    optimizer_parameters: tuple
    missing_trainable: tuple
    duplicate_parameters: tuple
    frozen_in_optimizer: tuple
    unknown_optimizer_parameters: tuple
    expected_trainable_not_enabled: tuple
    group_summaries: tuple
    frozen_parameters_are_errors: bool = False

    @property
    def passed(self):
        errors = (
            self.missing_trainable,
            self.duplicate_parameters,
            self.unknown_optimizer_parameters,
            self.expected_trainable_not_enabled,
        )
        if any(errors):
            return False
        return not (self.frozen_parameters_are_errors and self.frozen_in_optimizer)

    def raise_for_errors(self):
        if self.passed:
            return self
        details = []
        for field in (
            "missing_trainable",
            "duplicate_parameters",
            "unknown_optimizer_parameters",
            "expected_trainable_not_enabled",
        ):
            values = getattr(self, field)
            if values:
                details.append(f"{field}={list(values)}")
        if self.frozen_parameters_are_errors and self.frozen_in_optimizer:
            details.append(f"frozen_in_optimizer={list(self.frozen_in_optimizer)}")
        raise OptimizerAuditError("optimizer coverage audit failed: " + "; ".join(details))

    def as_dict(self):
        return {
            "passed": self.passed,
            "trainable_parameters": list(self.trainable_parameters),
            "optimizer_parameters": list(self.optimizer_parameters),
            "missing_trainable": list(self.missing_trainable),
            "duplicate_parameters": list(self.duplicate_parameters),
            "frozen_in_optimizer": list(self.frozen_in_optimizer),
            "unknown_optimizer_parameters": list(self.unknown_optimizer_parameters),
            "expected_trainable_not_enabled": list(self.expected_trainable_not_enabled),
            "group_summaries": list(self.group_summaries),
            "frozen_parameters_are_errors": self.frozen_parameters_are_errors,
        }


def audit_optimizer_coverage(
    model,
    optimizer,
    expected_trainable=(),
    fail_on_frozen=False,
):
    """Verify one-to-one coverage of every trainable model parameter."""

    model = getattr(model, "module", model)
    named_parameters = list(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in named_parameters}
    trainable_names = {
        name for name, parameter in named_parameters if parameter.requires_grad
    }
    frozen_names = {
        name for name, parameter in named_parameters if not parameter.requires_grad
    }

    counts = {}
    optimizer_names = set()
    unknown_names = []
    group_summaries = []
    for group_index, group in enumerate(optimizer.param_groups):
        group_trainable = 0
        group_frozen = 0
        parameters = list(group.get("params", ()))
        for parameter in parameters:
            parameter_id = id(parameter)
            counts[parameter_id] = counts.get(parameter_id, 0) + 1
            name = names_by_id.get(parameter_id)
            if name is None:
                name = f"<unknown:{len(unknown_names)}>"
                unknown_names.append(name)
            else:
                optimizer_names.add(name)
                if parameter.requires_grad:
                    group_trainable += 1
                else:
                    group_frozen += 1
        group_summaries.append(
            {
                "index": group_index,
                "num_parameters": len(parameters),
                "num_trainable": group_trainable,
                "num_frozen": group_frozen,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
            }
        )

    duplicate_names = []
    for parameter_id, count in counts.items():
        if count <= 1:
            continue
        duplicate_names.append(names_by_id.get(parameter_id, "<unknown-duplicate>"))
    expected_trainable = {str(name) for name in expected_trainable}
    report = OptimizerAuditReport(
        trainable_parameters=tuple(sorted(trainable_names)),
        optimizer_parameters=tuple(sorted(optimizer_names)),
        missing_trainable=tuple(sorted(trainable_names - optimizer_names)),
        duplicate_parameters=tuple(sorted(duplicate_names)),
        frozen_in_optimizer=tuple(sorted(frozen_names & optimizer_names)),
        unknown_optimizer_parameters=tuple(unknown_names),
        expected_trainable_not_enabled=tuple(sorted(expected_trainable - trainable_names)),
        group_summaries=tuple(group_summaries),
        frozen_parameters_are_errors=bool(fail_on_frozen),
    )
    return report
