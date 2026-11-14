"""The dense format: the flat positions the dense kernel scatters the values to."""

from torch import Tensor


def prepare_dense(shape: tuple[int, ...], index: Tensor, perm: Tensor) -> dict[str, Tensor]:
    """The format the dense kernel reads, derived from a checked pattern.

    Args:
        shape: the dense shape of the weight, outputs first. The format does not depend on it.
        index: ``(capacity,)`` int64 positions, -1 for a free slot.
        perm: ``(nnz,)`` the active slots ascending by position, from :func:`~spartorch.functional.sort_pattern`.

    Returns:
        ``perm``, and ``position``: ``(nnz,)`` int64, ``index[perm]``, the row-major flat
        position of each active slot. Nothing of the size of the dense weight.
    """
    return {"perm": perm, "position": index[perm]}
