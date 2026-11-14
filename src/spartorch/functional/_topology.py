"""Which connections exist: the checks and the order of a pattern."""

import torch
from torch import Tensor


def sort_pattern(index: Tensor, size: int) -> Tensor:
    r"""The active slots of a pattern, ascending by position.

    A pattern is ``index`` of shape ``(capacity,)``: ``index[s]`` is the row-major flat position of
    the connection in slot ``s`` in a dense weight of ``size`` entries, or -1 if the slot is free.
    The result ``perm`` of shape ``(nnz,)`` holds the slots with ``index[s] >= 0``, ordered so that
    ``index[perm]`` ascends. It reads the pattern and changes nothing.

    Args:
        index: ``(capacity,)``, int64.
        size: the number of entries of the dense weight, ``prod(shape)``.

    Returns:
        ``(nnz,)``, int64, on the device of ``index``.

    Raises:
        ValueError: unless ``index`` is 1-D int64, every active position is in :math:`[0, \mathrm{size})`
            and none appears twice.
    """
    if index.ndim != 1:
        raise ValueError(f"index must be a 1-D tensor, got {index.ndim}-D")
    if index.dtype != torch.int64:
        raise ValueError(f"index must be int64, got {index.dtype}")
    # Compared before anything is indexed: an out-of-range index on CUDA is a device-side assert
    # that ends the CUDA context of the whole process.
    if ((index < -1) | (index >= size)).any():
        raise ValueError(f"an active position is outside [0, {size})")
    slots = (index >= 0).nonzero().squeeze(1)
    position, order = index[slots].sort()
    if (position[1:] == position[:-1]).any():
        raise ValueError("a position appears twice")
    return slots[order]
