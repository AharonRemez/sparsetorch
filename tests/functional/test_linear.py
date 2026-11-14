import subprocess
import sys

import pytest
import torch

from spartorch import functional
from spartorch.functional import backends, sort_pattern
from spartorch.types import Backend
from tests._references import dense_linear

SHAPE = (4, 5)
# Slots 1 and 6 free, row 2 empty, slot 4 an active weight that is exactly 0.
INDEX = [0, -1, 4, 7, 9, 18, -1, 15, 6]
PATTERNS = {
    "pattern": (SHAPE, INDEX),
    "empty-pattern": (SHAPE, [-1, -1]),
    "no-outputs": ((0, 5), [-1]),
    "no-inputs": ((4, 0), [-1]),
}
# The leading dimensions of the input; the 3-D one is a strided view.
LEADING = {"1-D": (), "2-D": (3,), "3-D": (2, 3), "empty-batch": (0,)}


def pattern(device, dtype, backend="csr", shape=SHAPE, index=INDEX):
    index = torch.tensor(index, device=device)
    values = torch.randn(index.numel(), device=device, dtype=dtype)
    values[index < 0] = 5.0  # what a free slot holds must not reach the output
    values[4:5] = 0
    return index, values.requires_grad_(), functional.prepare(shape, index, backend=backend)


def strided(leading, n, device, dtype):
    if len(leading) < 2:
        return torch.randn(*leading, n, device=device, dtype=dtype)
    return torch.randn(*leading[:-1], n, leading[-1], device=device, dtype=dtype).transpose(-1, -2)


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("leading", LEADING.values(), ids=LEADING.keys())
@pytest.mark.parametrize(("shape", "index"), PATTERNS.values(), ids=PATTERNS.keys())
def test_forward_and_gradients_equal_the_dense_oracle(
    device, dtype, any_backend, close, shape, index, leading, with_bias
):
    index, values, fmt = pattern(device, dtype, any_backend, shape, index)
    out, in_ = shape
    x = strided(leading, in_, device, dtype).requires_grad_()
    assert x.is_contiguous() == (len(leading) < 2 or in_ == 0)
    bias = torch.randn(out, device=device, dtype=dtype, requires_grad=True) if with_bias else None
    leaves = [t for t in (values, x, bias) if t is not None]
    upstream = torch.randn(*leading, out, device=device, dtype=dtype)

    y = functional.linear(x, values, fmt, bias)
    expected = dense_linear(x, values, index, shape, bias)
    assert y.shape == expected.shape
    close(y, expected)
    grads = torch.autograd.grad(y, leaves, upstream)
    for got, want in zip(grads, torch.autograd.grad(expected, leaves, upstream), strict=True):
        close(got, want)
    assert grads[0][index < 0].eq(0).all()  # exactly: a free slot is not a connection
    if shape == SHAPE and index.numel() == len(INDEX):
        assert y[..., 2].eq(0 if bias is None else bias[2]).all()  # the empty row


def test_the_presets_agree_on_one_weight(device, dtype, close):
    index, values, _ = pattern(device, dtype)
    x = strided((2, 3), 5, device, dtype)
    csr, dense = (functional.linear(x, values, functional.prepare(SHAPE, index, backend=b)) for b in ("csr", "dense"))
    close(csr, dense)


def test_prepare_is_sort_pattern_then_the_backends_prepare(any_backend):
    index = torch.tensor(INDEX)
    fmt = functional.prepare(SHAPE, index, backend=any_backend)
    resolved = backends.resolve(any_backend)
    expected = resolved.prepare(SHAPE, index, sort_pattern(index, 20))
    assert (fmt.backend, fmt.shape, fmt.capacity) == (resolved, SHAPE, len(INDEX))
    assert fmt.buffers.keys() == expected.keys()
    assert all(torch.equal(fmt.buffers[name], expected[name]) for name in expected)


def test_the_default_backend_is_csr():
    assert functional.prepare(SHAPE, torch.tensor(INDEX)).backend is backends.resolve("csr")


def test_prepare_checks_the_pattern_against_the_shape():
    with pytest.raises(ValueError, match=r"outside \[0, 20\)"):
        functional.prepare(SHAPE, torch.tensor([20]))


def test_a_format_is_read_by_the_kernel_of_its_own_backend():
    calls = []

    def kernel(values, fmt, x, bias):
        calls.append(fmt)
        return x

    own = Backend("own", prepare=lambda shape, index, perm: {"perm": perm}, linear=kernel)
    fmt = functional.prepare(SHAPE, torch.tensor(INDEX), backend=own)
    assert fmt.backend is own
    assert fmt.buffers.keys() == {"perm"}
    x = torch.randn(3, 5)
    assert functional.linear(x, torch.zeros(len(INDEX)), fmt) is x
    assert calls == [fmt]


def test_a_backend_without_linear_is_named():
    fmt = functional.prepare(
        SHAPE, torch.tensor(INDEX), backend=Backend("bare", prepare=backends.resolve("dense").prepare)
    )
    with pytest.raises(ValueError, match="backend 'bare' has no linear"):
        functional.linear(torch.randn(3, 5), torch.zeros(len(INDEX)), fmt)


@pytest.mark.parametrize(
    ("x", "values", "bias", "match"),
    [
        ((3, 6), (9,), None, "input has 6 features, the weight has 5"),
        ((3, 5), (2,), None, r"values has shape \(2,\), the format has capacity 9"),
        ((3, 5), (9, 1), None, r"values has shape \(9, 1\), the format has capacity 9"),
        ((3, 5), (9,), (5,), r"bias has shape \(5,\), the weight has 4 outputs"),
        ((3, 5), (9,), (4, 1), r"bias has shape \(4, 1\), the weight has 4 outputs"),
    ],
    ids=["features", "short-values", "2-D-values", "bias-size", "2-D-bias"],
)
def test_arguments_that_do_not_fit_the_format(device, any_backend, x, values, bias, match):
    _, _, fmt = pattern(device, torch.float32, any_backend)
    with pytest.raises(ValueError, match=match):
        functional.linear(
            torch.randn(x, device=device),
            torch.randn(values, device=device),
            fmt,
            None if bias is None else torch.randn(bias, device=device),
        )


def test_the_dense_format_holds_nothing_of_the_dense_size():
    fmt = functional.prepare((1000, 1000), torch.tensor([3, -1, 900_000]), backend="dense")
    assert all(t.numel() <= fmt.capacity for t in fmt.buffers.values())


# The two calls the review of M0 reproduced as ending the CUDA context, before a Format carried
# the shape and the capacity with the buffers: a format of an (8, 8) weight used for a (4, 8)
# layer, and values shorter than the format.
ENDED_THE_CONTEXT = {
    "other-shape": "functional.linear(torch.randn(3, 8, device=d), torch.randn(64, device=d), "
    "functional.prepare((8, 8), torch.arange(64, device=d), backend=b), torch.randn(4, device=d))",
    "short-values": "functional.linear(torch.randn(3, 8, device=d), torch.randn(64, device=d)[:2], "
    "functional.prepare((8, 8), torch.arange(64, device=d), backend=b))",
}
CHILD = """
import torch
from spartorch import functional
d, b = {device!r}, {backend!r}
try:
    {call}
    print("computed")
except ValueError:
    print("refused")
try:
    torch.ones(2, device=d).add(1).sum().item()
    if d == "cuda":
        torch.cuda.synchronize()
    print("alive")
except Exception:
    print("dead")
"""


@pytest.mark.parametrize("call", ENDED_THE_CONTEXT.values(), ids=ENDED_THE_CONTEXT.keys())
def test_a_wrong_argument_is_refused_and_the_device_still_works(device, backend, call):
    # A child process: a device-side assert would end this process's CUDA context for every later test.
    child = subprocess.run(
        [sys.executable, "-c", CHILD.format(device=device, backend=backend, call=call)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.stdout.split() == ["refused", "alive"], child.stderr
