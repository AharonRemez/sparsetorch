r"""The CSR format: the compressed rows of the weight and of its transpose, what the CSR kernels read."""

import torch
import torch.nn.functional as F
from torch import Tensor


def prepare_csr(shape: tuple[int, ...], index: Tensor, perm: Tensor) -> dict[str, Tensor]:
    r"""The CSR format of a checked pattern, for :func:`~spartorch.functional.ops.linear_csr`.

    With ``position = index[perm]`` ascending, the active slots in row-major order are already
    in CSR order: entry :math:`k` of :math:`W` is slot ``perm[k]``, at row
    ``position[k] // in`` and column ``position[k] % in``. The pattern is not checked again.

    Args:
        shape: ``(out, in)``, the dense shape of the weight.
        index: ``(capacity,)`` int64 positions, -1 for a free slot.
        perm: ``(nnz,)`` the active slots ascending by position, from :func:`~spartorch.functional.sort_pattern`.

    Returns:
        int64 tensors on the device of ``index``, nothing of the size of the dense weight:
        ``perm``, ``(nnz,)``, the slot of each CSR entry of :math:`W`, so ``values[perm]`` is in
        CSR order; ``row_ptr``, ``(out + 1,)``, and ``col_idx``, ``(nnz,)``, the CSR of
        :math:`W`; ``t_perm``, ``(nnz,)``, for each CSR entry of :math:`W^\top` its place in
        ``values[perm]``; ``t_row_ptr``, ``(in + 1,)``, and ``t_col_idx``, ``(nnz,)``, the CSR
        of :math:`W^\top`.
    """
    n_rows, n_cols = shape
    position = index[perm]
    row, col = position // n_cols, position % n_cols
    t_perm = col.argsort(stable=True)  # CSR order is (row, col); stable by col gives (col, row)
    return {
        "perm": perm,
        "row_ptr": F.pad(torch.bincount(row, minlength=n_rows).cumsum(0), (1, 0)),
        "col_idx": col,
        "t_perm": t_perm,
        "t_row_ptr": F.pad(torch.bincount(col, minlength=n_cols).cumsum(0), (1, 0)),
        "t_col_idx": row[t_perm],
    }
