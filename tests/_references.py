"""Oracles written without any spartorch function."""

from math import prod

import torch.nn.functional as F
from torch import Tensor


def dense_linear(x: Tensor, values: Tensor, index: Tensor, shape: tuple[int, ...], bias: Tensor | None = None):
    """``F.linear`` on the dense weight whose flat position ``index[s]`` holds ``values[s]``; -1 is a free slot."""
    active = index >= 0
    weight = values.new_zeros(prod(shape))
    weight[index[active]] = values[active]
    return F.linear(x, weight.view(shape), bias)
