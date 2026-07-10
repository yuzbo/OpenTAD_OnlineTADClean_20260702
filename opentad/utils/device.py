from collections.abc import Mapping

import torch


def get_model_device(model):
    model = getattr(model, "module", model)
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def move_data_to_device(value, device, non_blocking=True):
    """Recursively move tensors while preserving metadata containers."""

    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)(
            (key, move_data_to_device(item, device, non_blocking))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(move_data_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [move_data_to_device(item, device, non_blocking) for item in value]
    return value
