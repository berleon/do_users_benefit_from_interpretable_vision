"""Pytorch helper functions."""

from __future__ import annotations

from typing import List, Optional, Sequence, TypeVar, Union

import torch
from torch import nn


def get_linear(
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    return x @ weight + bias


def dot4d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum('bchw,bchw->bhw', x, y)[:, None]


def view_as_grid(x: torch.Tensor, n_rows: int) -> torch.Tensor:
    b = x.shape[0]
    return x.view(b // n_rows, n_rows, *x.shape[1:])


T = TypeVar('T', bound=Union[torch.Tensor,
                             List[torch.Tensor],
                             Sequence[torch.Tensor]])


def move_to(
    obj: T,
    device: Union[torch.device, str],
) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)  # type: ignore
    elif isinstance(obj, list):
        return [move_to(x, device) for x in obj]  # type: ignore
    else:
        raise Exception()


def get_linear_layer(
        weight: torch.Tensor,
        bias: torch.Tensor,
        device: torch.device
) -> nn.Linear:
    if len(weight.shape) == 1:
        weight = weight[None, :]
    if len(bias.shape) == 0:
        bias = bias[None]
    linear = nn.Linear(len(weight), len(weight[0]))
    linear.weight.data = weight
    linear.bias.data = bias
    linear.to(device)
    return linear


class RecordOutputHook:
    def __init__(
        self,
        layer: nn.Module
    ):
        self.layer = layer
        self.hook = self.layer.register_forward_hook(self._callback)
        self._recorded_output: Optional[tuple[torch.Tensor, ...]] = None

    def detach(self):
        self.hook.remove()

    def _callback(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...]
    ):
        self._recorded_output = outputs

    @property
    def recorded_output(self) -> tuple[torch.Tensor, ...]:
        if self._recorded_output is None:
            raise ValueError("No output recorded")
        return self._recorded_output
