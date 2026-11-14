r"""The CSR kernels: the sparse products of a linear layer, through ``torch.sparse`` (cuSPARSE on CUDA).

A matrix :math:`A` of shape ``(n_rows, n_cols)`` is three arrays: ``values`` (floating,
``(nnz,)``), ``row_ptr`` (int64, ``(n_rows + 1,)``; the entries of row :math:`i` are
``row_ptr[i]:row_ptr[i + 1]``) and ``col_idx`` (int64, ``(nnz,)``, ascending inside a row).

Where the checks are: each op checks the dtypes, devices and shapes of its own arguments, from
metadata alone, because it is public and can be called without :func:`~spartorch.functional.linear`.
What only :func:`~spartorch.functional.linear` sees, the caller's tensors against the
:class:`~spartorch.types.Format` (``values`` against the capacity, the input and the bias against
the shape), is checked there and not here.
"""

import functools
from collections.abc import Callable
from math import prod
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import BackwardCFunction

from spartorch.types import Format

_SUPPORTED = {
    "cpu": (torch.float32, torch.float64),
    "cuda": (torch.float16, torch.bfloat16, torch.float32, torch.float64),
}


def _check_types(op: str, data: dict[str, Tensor], indices: dict[str, Tensor]) -> None:
    """ValueError unless ``data`` share a dtype ``op`` supports on their device, and ``indices`` are int64 there."""
    first, reference = next(iter(data.items()))
    if reference.dtype not in _SUPPORTED.get(reference.device.type, ()):
        raise ValueError(
            f"{op}: {first} is {reference.dtype}, which the CSR kernels do not support on "
            f"{reference.device.type}; they support {_SUPPORTED}"
        )
    for name, tensor in {**data, **indices}.items():
        dtype = reference.dtype if name in data else torch.int64
        if tensor.dtype != dtype or tensor.device != reference.device:
            raise ValueError(
                f"{op}: {name} is {tensor.dtype} on {tensor.device}, expected {dtype} on {reference.device}"
            )


def _check_shape(op: str, name: str, tensor: Tensor, shape: tuple[int, ...] | int) -> None:
    """ValueError unless ``tensor`` has ``shape``, or that many dimensions when it is an int."""
    if (tensor.dim() if isinstance(shape, int) else tuple(tensor.shape)) != shape:
        expected = f"{shape} dimensions" if isinstance(shape, int) else shape
        raise ValueError(f"{op}: {name} has shape {tuple(tensor.shape)}, expected {expected}")


def _documented[Op: Callable[..., Tensor]](op: Op) -> Op:
    # A registered op shows its class's docstring; this gives it its function's, and its signature.
    functools.update_wrapper(op, op._init_fn)  # pyright: ignore[reportFunctionMemberAccess] -- torch keeps the function under no public name
    return op


@_documented
@torch.library.custom_op("spartorch::spmm", mutates_args=())
def spmm(values: Tensor, row_ptr: Tensor, col_idx: Tensor, x: Tensor, n_rows: int) -> Tensor:
    r"""Sparse times dense: :math:`Y = X A^\top` for the CSR matrix :math:`A` = ``(values, row_ptr, col_idx)``.

    No autograd formula; :func:`~spartorch.functional.ops.linear_csr` is the differentiable form.
    On CUDA the result is not bitwise reproducible (cuSPARSE). Measured on 4096 x 4096 at 1 % with
    batch 128 and on 300 x 784 at 10 % with batch 32, over 50 calls in one process and across two
    processes, the largest difference was one unit in the last place of the largest output:
    1.1e-7 of :math:`\max |Y|` in fp32, 2.1e-16 in fp64.

    Runs in float32 and float64 on CPU, and also in float16 and bfloat16 on CUDA; another dtype
    raises ``ValueError`` naming it. Index values are not checked, because a check would cost a
    host sync per call: an index out of range, or row pointers that do not ascend from 0 to
    ``nnz``, is undefined behaviour on CPU and, on CUDA, a device-side assert that ends the CUDA
    context of the process. :func:`~spartorch.functional.sort_pattern` validates a pattern once,
    where it is built.

    Args:
        values: ``(nnz,)``, floating.
        row_ptr: ``(n_rows + 1,)``, int64.
        col_idx: ``(nnz,)``, int64, each in ``[0, n_cols)``; not checked.
        x: ``(batch, n_cols)``, the dtype and device of ``values``, any strides.
        n_rows: the number of rows of :math:`A`.

    Returns:
        ``(batch, n_rows)``, contiguous, with :math:`Y_{bi} = \sum_k \mathrm{values}_k\, X_{b,\mathrm{col\_idx}_k}`
        over the entries :math:`k` of row :math:`i`. A row without entries gives exactly 0.

    Raises:
        ValueError: for a dtype, device or shape that does not fit, before the kernel runs.
    """
    _check_types("spmm", {"values": values, "x": x}, {"row_ptr": row_ptr, "col_idx": col_idx})
    _check_shape("spmm", "x", x, 2)
    _check_shape("spmm", "values", values, 1)
    _check_shape("spmm", "col_idx", col_idx, tuple(values.shape))
    _check_shape("spmm", "row_ptr", row_ptr, (n_rows + 1,))
    # No torch invariant check: it reads the indices back from the device.
    a = torch.sparse_csr_tensor(row_ptr, col_idx, values, size=(n_rows, x.shape[1]), check_invariants=False)
    # cuSPARSE is 7-9x slower on a transposed view, and an op must not return a view.
    return torch.sparse.mm(a, x.t().contiguous()).t().contiguous()


@spmm.register_fake
def _(values: Tensor, row_ptr: Tensor, col_idx: Tensor, x: Tensor, n_rows: int) -> Tensor:
    return values.new_empty(x.shape[0], n_rows)


@_documented
@torch.library.custom_op("spartorch::sddmm", mutates_args=())
def sddmm(row_ptr: Tensor, col_idx: Tensor, g: Tensor, x: Tensor) -> Tensor:
    r"""Sampled dense-dense product: :math:`G^\top X` computed only at the positions of a CSR pattern.

    With :math:`Y = X A^\top` and :math:`G = \partial L / \partial Y` this is
    :math:`\partial L / \partial\,\mathrm{values}`, the weight gradient of a sparse linear layer,
    without forming the dense ``(n_rows, n_cols)`` product. Measured bitwise reproducible on CUDA,
    fp32 and fp64, over 50 calls in one process and across two processes, at the sizes of
    :func:`~spartorch.functional.ops.spmm`.

    Runs in float32 and float64 on CPU, and also in float16 and bfloat16 on CUDA; another dtype
    raises ``ValueError`` naming it. Index values are not checked, because a check would cost a
    host sync per call: an index out of range, or row pointers that do not ascend from 0 to
    ``nnz``, is undefined behaviour on CPU and, on CUDA, a device-side assert that ends the CUDA
    context of the process. :func:`~spartorch.functional.sort_pattern` validates a pattern once,
    where it is built.

    Args:
        row_ptr: ``(n_rows + 1,)``, int64.
        col_idx: ``(nnz,)``, int64, each in ``[0, n_cols)``; not checked.
        g: ``(batch, n_rows)``, floating, any strides.
        x: ``(batch, n_cols)``, the dtype and device of ``g``, any strides.

    Returns:
        ``(nnz,)`` in CSR order, contiguous, with :math:`\mathrm{out}_k = \sum_b G_{bi}\, X_{b,\mathrm{col\_idx}_k}`
        for the entry :math:`k` of row :math:`i`.

    Raises:
        ValueError: for a dtype, device or shape that does not fit, before the kernel runs.
    """
    _check_types("sddmm", {"g": g, "x": x}, {"row_ptr": row_ptr, "col_idx": col_idx})
    _check_shape("sddmm", "g", g, 2)
    _check_shape("sddmm", "x", x, 2)
    _check_shape("sddmm", "x", x, (g.shape[0], x.shape[1]))
    _check_shape("sddmm", "col_idx", col_idx, 1)
    _check_shape("sddmm", "row_ptr", row_ptr, (g.shape[1] + 1,))
    pattern = torch.sparse_csr_tensor(
        row_ptr, col_idx, g.new_zeros(col_idx.shape[0]), size=(g.shape[1], x.shape[1]), check_invariants=False
    )
    # 4-5x faster with a contiguous g.T and x stored column-major. Written into our own pattern:
    # without out=, sampled_addmm first copies its input, int64 column indices included.
    torch.sparse.sampled_addmm(pattern, g.t().contiguous(), x.t().contiguous().t(), beta=0.0, out=pattern)
    return pattern.values()


@sddmm.register_fake
def _(row_ptr: Tensor, col_idx: Tensor, g: Tensor, x: Tensor) -> Tensor:
    return g.new_empty(col_idx.shape[0])


@_documented
@torch.library.custom_op("spartorch::linear_csr", mutates_args=())
def linear_csr(
    values: Tensor,
    row_ptr: Tensor,
    col_idx: Tensor,
    t_perm: Tensor,
    t_row_ptr: Tensor,
    t_col_idx: Tensor,
    x: Tensor,
    n_rows: int,
) -> Tensor:
    r"""The differentiable sparse linear map: :math:`Y = X A^\top` for :math:`A` = ``(values, row_ptr, col_idx)``.

    The forward is :func:`~spartorch.functional.ops.spmm`. With :math:`G = \partial L / \partial Y`
    the backward is

    - :math:`\partial L / \partial\,\mathrm{values}` = :func:`~spartorch.functional.ops.sddmm` of
      ``(row_ptr, col_idx, g, x)``, that is :math:`G^\top X` at the pattern, and
    - :math:`\partial L / \partial X = G A` = :func:`~spartorch.functional.ops.spmm` of
      ``(values[t_perm], t_row_ptr, t_col_idx, g, n_cols)``, a product with the CSR of :math:`A^\top`,

    each computed only when that input requires a gradient. The index tensors get none. A second
    derivative is not supported and raises. On CUDA, measured, the output and the input gradient
    repeat as :func:`~spartorch.functional.ops.spmm` does, to one unit in the last place of the
    largest value, and the values gradient is bitwise reproducible, as
    :func:`~spartorch.functional.ops.sddmm` is.

    Runs in float32 and float64 on CPU, and also in float16 and bfloat16 on CUDA; another dtype
    raises ``ValueError`` naming it. Index values are not checked, because a check would cost a
    host sync per call: an index out of range, or row pointers that do not ascend from 0 to
    ``nnz``, is undefined behaviour on CPU and, on CUDA, a device-side assert that ends the CUDA
    context of the process. :func:`~spartorch.functional.sort_pattern` validates a pattern once,
    where it is built.

    Args:
        values: ``(nnz,)``, floating.
        row_ptr: ``(n_rows + 1,)``, int64.
        col_idx: ``(nnz,)``, int64, each in ``[0, n_cols)``; not checked.
        t_perm: ``(nnz,)``, int64: for each entry of :math:`A^\top` in its CSR order, its place in
            ``values``; not checked.
        t_row_ptr: ``(n_cols + 1,)``, int64, the row pointers of :math:`A^\top`.
        t_col_idx: ``(nnz,)``, int64, the column indices of :math:`A^\top`: rows of :math:`A`; not checked.
        x: ``(batch, n_cols)``, the dtype and device of ``values``, any strides.
        n_rows: the number of rows of :math:`A`.

    Returns:
        ``(batch, n_rows)``, contiguous.

    Raises:
        ValueError: for a dtype, device or shape that does not fit, before a kernel that reads it runs.
    """
    y = spmm(values, row_ptr, col_idx, x, n_rows)
    # After the forward: spmm has checked its arguments, x among them, and only the backward
    # reads the transpose.
    _check_types("linear_csr", {"values": values}, {"t_perm": t_perm, "t_row_ptr": t_row_ptr, "t_col_idx": t_col_idx})
    _check_shape("linear_csr", "t_perm", t_perm, tuple(values.shape))
    _check_shape("linear_csr", "t_col_idx", t_col_idx, tuple(values.shape))
    _check_shape("linear_csr", "t_row_ptr", t_row_ptr, (x.shape[1] + 1,))
    return y


@linear_csr.register_fake
def _(
    values: Tensor,
    row_ptr: Tensor,
    col_idx: Tensor,
    t_perm: Tensor,
    t_row_ptr: Tensor,
    t_col_idx: Tensor,
    x: Tensor,
    n_rows: int,
) -> Tensor:
    return values.new_empty(x.shape[0], n_rows)


def _linear_csr_setup(ctx: BackwardCFunction, inputs: tuple[Any, ...], output: Tensor) -> None:
    ctx.save_for_backward(*inputs[:-1])


def _linear_csr_backward(ctx: Any, g: Tensor) -> tuple[Tensor | None, ...]:  # noqa: ANN401 -- torch's stub types saved_tensors and needs_input_grad as 1-tuples
    values, row_ptr, col_idx, t_perm, t_row_ptr, t_col_idx, x = ctx.saved_tensors
    grad_values = grad_x = None
    if ctx.needs_input_grad[0]:
        grad_values = sddmm(row_ptr, col_idx, g, x)
    if ctx.needs_input_grad[6]:
        # g @ A as an spmm on the cached transpose: torch's own transposed paths re-convert on
        # every call and break CUDA-graph capture.
        grad_x = spmm(values[t_perm], t_row_ptr, t_col_idx, g, x.shape[1])
    # A second derivative raises: spmm and sddmm have no autograd formula.
    return grad_values, None, None, None, None, None, grad_x, None


linear_csr.register_autograd(_linear_csr_backward, setup_context=_linear_csr_setup)


def linear_csr_format(values: Tensor, format: Format, x: Tensor, bias: Tensor | None = None) -> Tensor:
    r""":math:`y = x W^\top + \mathrm{bias}` with ``W`` read from the CSR format: the linear of the ``"csr"`` backend.

    The values are gathered into CSR order, ``values[perm]``, and every leading dimension of ``x``
    becomes one batch dimension of :func:`~spartorch.functional.ops.linear_csr`, so the gradients
    are its own: :func:`~spartorch.functional.ops.sddmm` for ``values`` (free slots get exactly
    0), the transposed product for ``x``, each only when it is needed. The kernels do not promote
    and do not run under autocast: ``x`` must have the dtype of ``values``, float32 or float64 on
    CPU and also float16 or bfloat16 on CUDA.

    Args:
        values: ``(capacity,)``, floating.
        format: its :attr:`~spartorch.types.Format.buffers` are those of
            :func:`~spartorch.functional.storage.prepare_csr`, for ``format.shape = (out, in)``.
        x: ``(..., in)``, the dtype and device of ``values``.
        bias: ``(out,)`` or None.

    Returns:
        ``(..., out)``.

    Raises:
        ValueError: if ``x`` and ``values`` differ in dtype, before a kernel runs, or for a dtype
            or device the kernels refuse.
    """
    if x.dtype != values.dtype:
        raise ValueError(
            f"backend {format.backend.name!r}: the input is {x.dtype} and the values are {values.dtype}; "
            "the CSR kernels do not promote, so autocast is not supported on this backend"
        )
    b = format.buffers
    out = format.shape[0]
    # index_select, not values[perm]: its backward is an index_add, not a sort-based index_put.
    # Not reshape(-1, in): it cannot infer the batch when in is 0.
    y = linear_csr(
        values.index_select(0, b["perm"]),
        b["row_ptr"],
        b["col_idx"],
        b["t_perm"],
        b["t_row_ptr"],
        b["t_col_idx"],
        x.reshape(prod(x.shape[:-1]), x.shape[-1]),
        out,
    ).view(*x.shape[:-1], out)
    return y if bias is None else y + bias
