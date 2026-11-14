"""The dense kernel: the sparse weight scattered into a dense one."""

from math import prod

import torch.nn.functional as F
from torch import Tensor

from spartorch.types import Format


def linear_dense(values: Tensor, format: Format, x: Tensor, bias: Tensor | None = None) -> Tensor:
    r""":math:`y = x W^\top + \mathrm{bias}` with ``W`` scattered from the slots.

    ``W`` is ``(out, in) = format.shape``, 0 outside the pattern, with ``values[perm[k]]`` at row-major
    flat position ``position[k]``. The product is ``F.linear``: native autograd gives every
    gradient, and free slots get gradient 0.

    Args:
        values: ``(capacity,)``, floating.
        format: its :attr:`~spartorch.types.Format.buffers` are ``perm`` and ``position``, ``(nnz,)`` each, from
            :func:`~spartorch.functional.storage.prepare_dense`.
        x: ``(..., in)``.
        bias: ``(out,)`` or None.

    Returns:
        ``(..., out)``.
    """
    # In place on the fresh zeros: the out-of-place index_put copies the whole weight first.
    # index_select, not values[perm]: its backward is an index_add, not a sort-based index_put.
    buffers = format.buffers
    weight = values.new_zeros(prod(format.shape)).index_put_(
        (buffers["position"],), values.index_select(0, buffers["perm"])
    )
    return F.linear(x, weight.view(format.shape), bias)
