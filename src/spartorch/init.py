"""How the values of a sparse weight start, as ``torch.nn.init``."""

from math import prod

import torch
from torch import Tensor


@torch.no_grad()
def fan_in_uniform_(values: Tensor, index: Tensor, shape: tuple[int, ...]) -> Tensor:
    r"""Draw every active weight of output row ``i`` from :math:`U(-1/\sqrt{k_i}, 1/\sqrt{k_i})`, in place.

    :math:`k_i` is the number of connections of row ``i``, the sparse fan-in of output ``i`` (Evci et
    al., arXiv 2010.03533): ``nn.Linear``'s default distribution with the actual fan-in in place of
    ``in_features``, so a weight has mean 0 and variance :math:`1 / (3 k_i)`. Row ``i`` is
    ``index // prod(shape[1:])``. Free slots are not touched; rows without connections have
    nothing to draw. The draw uses the global torch generator on the device of ``values``, keeps
    their dtype, and records no gradient. The bound :math:`1/\sqrt{k_i}` is computed in at least
    fp32, so in half precision a row of more than 65,504 connections still gets its own.

    Args:
        values: ``(capacity,)``, floating; written in place.
        index: ``(capacity,)``, int64 row-major positions in ``shape``, -1 for a free slot, as
            :func:`~spartorch.functional.sort_pattern` accepts.
        shape: the dense shape of the weight, outputs first.

    Returns:
        ``values``.
    """
    active = index >= 0
    row = index[active] // prod(shape[1:])
    # The count in a dtype that holds it: in fp16 a count past 65,504 is inf, and rsqrt(inf) is 0.
    bound = torch.bincount(row)[row].to(torch.promote_types(values.dtype, torch.float32)).rsqrt().to(values.dtype)
    values[active] = (torch.rand_like(bound) * 2 - 1) * bound
    return values
