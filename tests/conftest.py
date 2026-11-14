from typing import TypedDict

import pytest
import torch
import torch.nn.functional as F

from spartorch.functional.storage import prepare_csr
from spartorch.types import Backend


# torch.sparse.mm on CUDA is not bitwise reproducible (about 1e-5 in fp32): always tolerances.
class Tol(TypedDict):
    rtol: float
    atol: float


TOL: dict[torch.dtype, Tol] = {
    torch.float32: Tol(rtol=1e-4, atol=1e-5),
    torch.float64: Tol(rtol=1e-10, atol=1e-12),
    # One rounding of the result: the CSR kernels accumulate in fp32, and their error against an
    # fp64 oracle measured half an ulp; one ulp relative plus one absolute covers it twice over.
    torch.float16: Tol(rtol=1e-3, atol=1e-3),
    torch.bfloat16: Tol(rtol=8e-3, atol=8e-3),
}


def pytest_collection_modifyitems(items):
    if torch.cuda.is_available():
        return
    for item in items:
        if item.get_closest_marker("cuda"):
            item.add_marker(pytest.mark.skip(reason="CUDA is not available"))


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def dtype(request):
    return request.param


PRESETS = ["csr", "dense"]


@pytest.fixture(params=PRESETS)
def backend(request):
    """A preset name, for the tests that pass the backend to a child process or are about the presets."""
    return request.param


def _csr_read_densely(values, format, x, bias):
    # Another kernel on the csr format, written with torch alone: the row of entry k is the last
    # row whose pointer is at most k.
    b = format.buffers
    row = torch.searchsorted(b["row_ptr"], torch.arange(b["col_idx"].numel(), device=x.device), right=True) - 1
    weight = values.new_zeros(format.shape).index_put_((row, b["col_idx"]), values[b["perm"]])
    return F.linear(x, weight, bias)


CUSTOM = Backend("csr-read-densely", prepare=prepare_csr, linear=_csr_read_densely)


@pytest.fixture(params=[*PRESETS, CUSTOM], ids=[*PRESETS, CUSTOM.name])
def any_backend(request):
    """The two presets and a backend written here: what spartorch computes must not depend on which."""
    return request.param


@pytest.fixture
def close():
    return lambda actual, expected: torch.testing.assert_close(actual, expected, **TOL[expected.dtype])
