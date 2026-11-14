import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from spartorch import functional
from spartorch.functional.ops import _csr, linear_csr, linear_csr_format, sddmm, spmm
from tests._references import dense_linear
from tests.conftest import TOL


def csr(position: Tensor, shape: tuple[int, int]) -> tuple[Tensor, ...]:
    """``(row_ptr, col_idx, t_perm, t_row_ptr, t_col_idx)`` of ascending flat positions, with torch alone."""
    n_rows, n_cols = shape
    row, col = position // n_cols, position % n_cols
    t_perm = torch.argsort(col * n_rows + row)
    return (
        F.pad(torch.bincount(row, minlength=n_rows).cumsum(0), (1, 0)),
        col,
        t_perm,
        F.pad(torch.bincount(col, minlength=n_cols).cumsum(0), (1, 0)),
        row[t_perm],
    )


# Rows 0, 3 and the last one empty; one whole row; no entry at all.
PATTERNS = {"empty-rows": (6, 9), "single-row": (1, 9), "empty-matrix": (3, 9)}


@pytest.fixture(params=PATTERNS.values(), ids=list(PATTERNS))
def case(request, device, dtype):
    """A pattern as ``(shape, position, values, arrays)``; the second weight, if any, is exactly 0."""
    shape = request.param
    keep = torch.rand(shape, device=device) < 0.5
    if shape == (6, 9):
        keep[[0, 3, 5]] = False
    elif shape == (1, 9):
        keep[:] = True
    else:
        keep[:] = False
    position = keep.flatten().nonzero().squeeze(1)
    values = torch.randn(position.numel(), device=device, dtype=dtype)
    values[1:2] = 0
    return shape, position, values, csr(position, shape)


@pytest.fixture(params=["contiguous", "strided", "batch-of-one"])
def batch(request, device, dtype):
    """A dense ``(batch, n)`` factory: contiguous, a transposed view, or one row."""

    def make(n: int) -> Tensor:
        if request.param == "strided":
            return torch.randn(n, 4, device=device, dtype=dtype).t()  # not contiguous unless n is 1
        return torch.randn(1 if request.param == "batch-of-one" else 4, n, device=device, dtype=dtype)

    return make


def test_spmm_equals_the_dense_oracle(case, batch, close):
    shape, position, values, (row_ptr, col_idx, *_) = case
    x = batch(shape[1])
    y = spmm(values, row_ptr, col_idx, x, shape[0])
    close(y, dense_linear(x, values, position, shape))
    assert y.is_contiguous()
    assert y[:, row_ptr.diff() == 0].eq(0).all()  # an empty row is exactly 0


def test_sddmm_equals_the_weight_gradient_of_the_dense_oracle(case, batch, close):
    shape, position, values, (row_ptr, col_idx, *_) = case
    x, g = batch(shape[1]), batch(shape[0])
    v = values.clone().requires_grad_()
    (expected,) = torch.autograd.grad(dense_linear(x, v, position, shape), v, g)
    out = sddmm(row_ptr, col_idx, g, x)
    close(out, expected)
    assert out.is_contiguous()


def test_linear_csr_forward_and_gradients_equal_the_dense_oracle(case, batch, close):
    shape, position, values, arrays = case
    values = values.clone().requires_grad_()
    x, g = batch(shape[1]).requires_grad_(), batch(shape[0])
    y = linear_csr(values, *arrays, x, shape[0])
    expected = dense_linear(x, values, position, shape)
    close(y, expected)
    grads = torch.autograd.grad(y, (values, x), g)
    for got, want in zip(grads, torch.autograd.grad(expected, (values, x), g), strict=True):
        close(got, want)
    if values.numel() > 1:
        assert grads[0][1] != 0  # the weight that is exactly 0 is in the pattern and learns


@pytest.mark.parametrize("dtype", [torch.float64], ids=["fp64"])
def test_linear_csr_gradcheck(case):
    shape, _, values, arrays = case
    values = values.clone().requires_grad_()
    x = torch.randn(4, shape[1], device=values.device, dtype=values.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v, x: linear_csr(v, *arrays, x, shape[0]), (values, x))


def test_opcheck(case):
    shape, _, values, (row_ptr, col_idx, *transpose) = case
    x = torch.randn(4, shape[1], device=values.device, dtype=values.dtype)
    g = torch.randn(4, shape[0], device=values.device, dtype=values.dtype)
    tol = TOL[values.dtype]
    torch.library.opcheck(spmm, (values, row_ptr, col_idx, x, shape[0]), **tol)
    torch.library.opcheck(sddmm, (row_ptr, col_idx, g, x), **tol)
    args = (values.clone().requires_grad_(), row_ptr, col_idx, *transpose, x.requires_grad_(), shape[0])
    torch.library.opcheck(linear_csr, args, **tol)


def test_one_example_gets_the_input_gradient_of_its_own_loss_only(device, dtype):
    shape = (6, 9)
    position = torch.tensor([1, 5, 12, 20, 33, 47, 53], device=device)
    values = torch.randn(position.numel(), device=device, dtype=dtype)
    x = torch.randn(5, 9, device=device, dtype=dtype, requires_grad=True)
    linear_csr(values, *csr(position, shape), x, 6)[2].sum().backward()
    assert x.grad is not None
    assert x.grad[2].ne(0).any()
    assert x.grad[[0, 1, 3, 4]].eq(0).all()


def small(device, dtype=torch.float32):
    """A valid call of each op on a (3, 5) pattern of 4 entries, as keyword arguments."""
    row_ptr, col_idx, t_perm, t_row_ptr, t_col_idx = csr(torch.tensor([1, 4, 5, 13], device=device), (3, 5))
    values = torch.randn(4, device=device, dtype=dtype)
    x, g = torch.randn(2, 5, device=device, dtype=dtype), torch.randn(2, 3, device=device, dtype=dtype)
    transpose = {"t_perm": t_perm, "t_row_ptr": t_row_ptr, "t_col_idx": t_col_idx}
    return {
        "spmm": {"values": values, "row_ptr": row_ptr, "col_idx": col_idx, "x": x, "n_rows": 3},
        "sddmm": {"row_ptr": row_ptr, "col_idx": col_idx, "g": g, "x": x},
        "linear_csr": {"values": values, "row_ptr": row_ptr, "col_idx": col_idx, **transpose, "x": x, "n_rows": 3},
    }


OPS = {"spmm": spmm, "sddmm": sddmm, "linear_csr": linear_csr}


def test_backward_runs_only_the_kernels_it_needs(device, monkeypatch):
    # sddmm for the weight gradient, spmm for the input gradient, and neither for a tensor that
    # does not require one: the first layer of a network never pays for the input gradient.
    calls = []
    for name in ("spmm", "sddmm"):
        kernel = getattr(_csr, name)
        monkeypatch.setattr(_csr, name, lambda *a, _name=name, _kernel=kernel: calls.append(_name) or _kernel(*a))
    kwargs = small(device)["linear_csr"]
    for values_grad, x_grad, expected in (
        (True, False, ["sddmm"]),
        (False, True, ["spmm"]),
        (True, True, ["sddmm", "spmm"]),
    ):
        values = kwargs["values"].clone().requires_grad_(values_grad)
        x = kwargs["x"].clone().requires_grad_(x_grad)
        y = linear_csr(**{**kwargs, "values": values, "x": x})
        calls.clear()  # the forward is one spmm
        y.sum().backward()
        assert sorted(calls) == expected


@pytest.mark.parametrize("name", OPS)
def test_an_op_shows_the_docstring_and_signature_of_its_function(name):
    # What the API page and help() show: a registered op is otherwise documented as its class.
    assert "Args:" in (OPS[name].__doc__ or "")
    assert list(inspect.signature(OPS[name]).parameters) == list(small("cpu")[name])


def test_a_second_derivative_raises(device):
    kwargs = small(device, torch.float64)["linear_csr"]
    values, x = kwargs["values"].requires_grad_(), kwargs["x"].requires_grad_()
    y = linear_csr(**kwargs)
    for first, second in ((values, x), (x, values)):
        (grad,) = torch.autograd.grad(y.sum(), first, create_graph=True)
        with pytest.raises(RuntimeError, match="no autograd formula"):
            torch.autograd.grad(grad.sum(), second)


@pytest.mark.parametrize("name", OPS)
def test_the_output_is_contiguous_and_aliases_no_input(device, name):
    kwargs = small(device)[name]
    out = OPS[name](**kwargs)
    assert out.is_contiguous()
    inputs = [t.untyped_storage().data_ptr() for t in kwargs.values() if isinstance(t, Tensor)]
    assert out.untyped_storage().data_ptr() not in inputs


def changed(device, name, argument, change):
    kwargs = small(device)[name]
    return {**kwargs, argument: change(kwargs[argument])}


def int32(t):
    return t.int()


# Each is a call a kernel may not survive, or one whose error would otherwise be the backend's.
WRONG = {
    "spmm-x-1-D": ("spmm", "x", lambda t: t[0], r"spmm: x has shape \(5,\), expected 2 dimensions"),
    "spmm-values-2-D": ("spmm", "values", lambda t: t[:, None], r"spmm: values has shape \(4, 1\), expected 1 dim"),
    "spmm-col_idx-short": ("spmm", "col_idx", lambda t: t[:3], r"spmm: col_idx has shape \(3,\), expected \(4,\)"),
    "spmm-row_ptr-short": ("spmm", "row_ptr", lambda t: t[:3], r"spmm: row_ptr has shape \(3,\), expected \(4,\)"),
    "spmm-n_rows": ("spmm", "n_rows", lambda n: n + 1, r"spmm: row_ptr has shape \(4,\), expected \(5,\)"),
    "spmm-x-dtype": ("spmm", "x", lambda t: t.double(), "spmm: x is torch.float64 on .*, expected torch.float32"),
    "spmm-values-int": ("spmm", "values", lambda t: t.long(), "spmm: values is torch.int64, which the CSR kernels"),
    "spmm-values-complex": ("spmm", "values", lambda t: t.cfloat(), "spmm: values is torch.complex64, which"),
    "spmm-row_ptr-int32": ("spmm", "row_ptr", int32, "spmm: row_ptr is torch.int32 on .*, expected torch.int64"),
    "spmm-col_idx-int32": ("spmm", "col_idx", int32, "spmm: col_idx is torch.int32 on .*, expected torch.int64"),
    "sddmm-g-1-D": ("sddmm", "g", lambda t: t[0], r"sddmm: g has shape \(3,\), expected 2 dimensions"),
    "sddmm-x-1-D": ("sddmm", "x", lambda t: t[0], r"sddmm: x has shape \(5,\), expected 2 dimensions"),
    "sddmm-x-batch": ("sddmm", "x", lambda t: t[:1], r"sddmm: x has shape \(1, 5\), expected \(2, 5\)"),
    "sddmm-g-rows": ("sddmm", "g", lambda t: t[:, :2], r"sddmm: row_ptr has shape \(4,\), expected \(3,\)"),
    "sddmm-col_idx-2-D": ("sddmm", "col_idx", lambda t: t[:, None], r"sddmm: col_idx has shape \(4, 1\), expected 1"),
    "sddmm-x-dtype": ("sddmm", "x", lambda t: t.double(), "sddmm: x is torch.float64 on .*, expected torch.float32"),
    "sddmm-g-int": ("sddmm", "g", lambda t: t.long(), "sddmm: g is torch.int64, which the CSR kernels"),
    "sddmm-row_ptr-int32": ("sddmm", "row_ptr", int32, "sddmm: row_ptr is torch.int32 on .*, expected torch.int64"),
    "sddmm-col_idx-int32": ("sddmm", "col_idx", int32, "sddmm: col_idx is torch.int32 on .*, expected torch.int64"),
    "linear-forward": ("linear_csr", "col_idx", lambda t: t[:3], r"spmm: col_idx has shape \(3,\), expected \(4,\)"),
    "linear-t_perm-short": ("linear_csr", "t_perm", lambda t: t[:3], r"linear_csr: t_perm has shape \(3,\), exp"),
    "linear-t_col_idx-short": ("linear_csr", "t_col_idx", lambda t: t[:3], r"linear_csr: t_col_idx has shape \(3,"),
    "linear-t_row_ptr-short": (
        "linear_csr",
        "t_row_ptr",
        lambda t: t[:5],
        r"t_row_ptr has shape \(5,\), expected \(6,",
    ),
    "linear-t_perm-int32": ("linear_csr", "t_perm", int32, "linear_csr: t_perm is torch.int32"),
    "linear-t_row_ptr-int32": ("linear_csr", "t_row_ptr", int32, "linear_csr: t_row_ptr is torch.int32"),
    "linear-t_col_idx-int32": ("linear_csr", "t_col_idx", int32, "linear_csr: t_col_idx is torch.int32"),
}


@pytest.mark.parametrize(("name", "argument", "change", "match"), WRONG.values(), ids=WRONG.keys())
def test_a_wrong_argument_raises_before_the_kernel(device, name, argument, change, match):
    with pytest.raises(ValueError, match=match):
        OPS[name](**changed(device, name, argument, change))


@pytest.mark.parametrize("name", OPS)
def test_index_values_are_not_read_back(name):
    # torch's invariant check would read the indices from the device on every call: columns out
    # of order inside a row, in range and harmless to the kernel, show that it is off.
    kwargs = small("cpu")[name]
    kwargs["col_idx"] = kwargs["col_idx"].flip(0)  # (3, 5) at [1, 4, 5, 13]: row 0 holds columns 1 and 4
    OPS[name](**kwargs)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("name", "argument"),
    [("spmm", "x"), ("spmm", "row_ptr"), ("sddmm", "x"), ("sddmm", "col_idx"), ("linear_csr", "t_perm")],
)
def test_a_tensor_on_another_device_raises(name, argument):
    with pytest.raises(ValueError, match=f"{name}: {argument} is torch.* on cpu, expected .* on cuda:0"):
        OPS[name](**changed("cuda", name, argument, lambda t: t.cpu()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("name", OPS)
def test_half_precision_is_refused_on_cpu_by_name(name, dtype):
    checked_by = "spmm" if name == "linear_csr" else name  # its forward checks the forward's arguments
    with pytest.raises(ValueError, match=f"{checked_by}: .* is {dtype}, which the CSR kernels do not support on cpu"):
        OPS[name](**small("cpu", dtype)[name])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_half_precision_on_cuda_equals_the_dense_oracle(dtype, close):
    # Large enough that the rounding of a long sum shows: 64 x 256 at 20 %, batch 32.
    shape = (64, 256)
    position = torch.randperm(64 * 256, device="cuda")[: 64 * 256 // 5].sort().values
    values = torch.randn(position.numel(), device="cuda", dtype=dtype, requires_grad=True)
    x = torch.randn(32, 256, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(32, 64, device="cuda", dtype=dtype)
    y = linear_csr(values, *csr(position, shape), x, 64)
    grads = torch.autograd.grad(y, (values, x), g)
    # The oracle in fp64 from the same half inputs, rounded once to the dtype.
    v64, x64 = values.detach().double().requires_grad_(), x.detach().double().requires_grad_()
    expected = dense_linear(x64, v64, position, shape)
    close(y, expected.to(dtype))
    for got, want in zip(grads, torch.autograd.grad(expected, (v64, x64), g.double()), strict=True):
        close(got, want.to(dtype))


@pytest.mark.parametrize("dtype", [torch.float32], ids=["fp32"])
def test_compiled_linear_csr_equals_eager(device, dtype, close):
    kwargs = small(device, dtype)["linear_csr"]
    values, x = kwargs["values"].requires_grad_(), kwargs["x"].requires_grad_()

    def loss(values, x):
        return linear_csr(**{**kwargs, "values": values, "x": x}).square().sum()

    compiled = torch.compile(loss, fullgraph=True)
    for got, want in zip(
        (compiled(values, x), *torch.autograd.grad(compiled(values, x), (values, x))),
        (loss(values, x), *torch.autograd.grad(loss(values, x), (values, x))),
        strict=True,
    ):
        close(got, want)


def layer(dtype):
    """4096 x 4096 at 1 %, batch 128, on CUDA: as ``(values, row_ptr, col_idx, x, g)``."""
    n = 4096
    # The permutation on the CPU: on CUDA it would leave ~0.4 GB in the cache of every test process.
    position = torch.randperm(n * n)[: n * n // 100].sort().values.cuda()
    row_ptr, col_idx, *_ = csr(position, (n, n))
    values = torch.randn(position.numel(), device="cuda", dtype=dtype)
    return (
        values,
        row_ptr,
        col_idx,
        torch.randn(128, n, device="cuda", dtype=dtype),
        torch.randn(128, n, device="cuda", dtype=dtype),
    )


# The claims of the docstrings, loosened about 8 times from what was measured.
SPMM_SPREAD = {torch.float32: 1e-6, torch.float64: 2e-15}  # measured 1.1e-7 and 2.1e-16


@pytest.mark.cuda
def test_sddmm_is_bitwise_reproducible(dtype):
    _, row_ptr, col_idx, x, g = layer(dtype)
    first = sddmm(row_ptr, col_idx, g, x)
    assert all(torch.equal(sddmm(row_ptr, col_idx, g, x), first) for _ in range(10))


@pytest.mark.cuda
def test_spmm_repeats_within_its_stated_bound(dtype):
    values, row_ptr, col_idx, x, _ = layer(dtype)
    first = spmm(values, row_ptr, col_idx, x, 4096)
    spread = max((spmm(values, row_ptr, col_idx, x, 4096) - first).abs().max().item() for _ in range(10))
    assert spread <= SPMM_SPREAD[dtype] * first.abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16, torch.int64])
def test_the_format_linear_names_a_dtype_mismatch_before_a_kernel_runs(device, dtype, monkeypatch):
    # The kernels do not promote; under autocast a mismatch is what a user of a layer meets, and
    # the kernel's own message names its arguments, not theirs.
    monkeypatch.setattr(_csr, "linear_csr", None)
    fmt = functional.prepare((3, 4), torch.tensor([0, 5], device=device), backend="csr")
    x = torch.ones(2, 4, device=device, dtype=dtype)
    match = (
        f"backend 'csr': the input is {dtype} and the values are torch.float32; the CSR kernels do not "
        "promote, so autocast is not supported on this backend"
    )
    with pytest.raises(ValueError, match=match):
        linear_csr_format(torch.ones(2, device=device), fmt, x)
