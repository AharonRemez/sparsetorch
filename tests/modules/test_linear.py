import copy
from itertools import pairwise

import pytest
import torch
from torch import nn

from spartorch import SparseLinear, SparseWeight, functional
from spartorch.functional.ops import _csr
from spartorch.init import fan_in_uniform_
from tests._references import dense_linear
from tests.conftest import TOL


def oracle(layer, x):
    w = layer.weight
    return dense_linear(x, w.values, w.index, w.shape, layer.bias)


def test_forward_is_functional_linear(device, any_backend):
    layer = SparseLinear(6, 5, density=0.4, capacity=15, backend=any_backend).to(device)
    x = torch.randn(2, 3, 6, device=device)
    w = layer.weight
    assert torch.equal(layer(x), functional.linear(x, w.values, w.format(), layer.bias))


@pytest.mark.parametrize(
    ("kwargs", "nnz"), [({"density": 0.25}, 15), ({"nnz": 0}, 0), ({"nnz": 7}, 7), ({"nnz": 60}, 60)]
)
def test_number_of_connections(kwargs, nnz):
    layer = SparseLinear(10, 6, capacity=60, bias=False, **kwargs)
    assert int((layer.weight.index >= 0).sum()) == nnz
    assert layer.weight.values.numel() == 60
    assert layer.bias is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one of density and nnz"),
        ({"density": 0.5, "nnz": 3}, "exactly one of density and nnz"),
        ({"nnz": 61}, r"outside \[0, 60\]"),
        ({"nnz": 5, "capacity": 4}, "capacity 4 is below"),
    ],
)
def test_rejects_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SparseLinear(10, 6, **kwargs)


def test_reset_parameters_is_fan_in_uniform():
    layer = SparseLinear(8, 4, density=0.5, capacity=20)
    w = layer.weight
    torch.manual_seed(1)
    layer.reset_parameters()
    torch.manual_seed(1)
    expected = fan_in_uniform_(torch.zeros(20), w.index, (4, 8))
    assert torch.equal(w.values.detach(), expected)


def test_bias_is_uniform_within_one_over_sqrt_in_features():
    bias = SparseLinear(16, 1000, nnz=10).bias
    assert bias is not None
    assert bias.abs().max() <= 0.25
    # 1000 draws from U(-0.25, 0.25) all above -0.2 (or all below 0.2): probability 0.9^1000
    assert bias.min() < -0.2
    assert bias.max() > 0.2
    zero = SparseLinear(0, 3, nnz=0).bias
    assert zero is not None
    assert not zero.any()


def test_sizes_are_read_from_the_weight():
    layer = SparseLinear(6, 5, nnz=4)
    assert (layer.in_features, layer.out_features) == (6, 5)
    layer.weight = SparseWeight((7, 3), torch.tensor([0, 20]))
    assert (layer.in_features, layer.out_features) == (3, 7)


def test_the_default_backend_is_csr():
    assert SparseLinear(4, 3, nnz=5).weight.backend is functional.backends.resolve("csr")


def test_state_dict_keys(any_backend):
    assert list(SparseLinear(4, 3, nnz=5, backend=any_backend).state_dict()) == [
        "bias",
        "weight.values",
        "weight.index",
        "weight.shape",
    ]


def test_sparse_mlp_trains(device, any_backend):
    x = torch.randn(512, 16, device=device)
    y = torch.sin(x[:, :4].sum(1, keepdim=True)) + x[:, 4:5] * x[:, 5:6]
    model = nn.Sequential(
        SparseLinear(16, 64, density=0.3, backend=any_backend),
        nn.ReLU(),
        SparseLinear(64, 1, density=0.3, backend=any_backend),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(300):
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2


HALF = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", HALF, ids=["fp16", "bf16"])
def test_half_precision(device, backend, dtype):
    # The kernels' table, met through a layer: csr runs 16-bit on CUDA only, dense everywhere.
    layer = SparseLinear(256, 64, density=0.2, backend=backend).to(device, dtype)
    x = torch.randn(32, 256, device=device, dtype=dtype, requires_grad=True)
    if (backend, device) == ("csr", "cpu"):
        with pytest.raises(ValueError, match=f"is {dtype}, which the CSR kernels do not support on cpu"):
            layer(x)
        return
    y = layer(x)
    g = torch.randn_like(y)
    leaves = [layer.weight.values, x, layer.bias]
    grads = torch.autograd.grad(y, leaves, g)
    # The oracle in fp64 from the same 16-bit tensors, rounded once to the dtype.
    wide = [t.detach().double().requires_grad_() for t in leaves]
    w = layer.weight
    expected = dense_linear(wide[1], wide[0], w.index, w.shape, wide[2])
    torch.testing.assert_close(y, expected.to(dtype), **TOL[dtype])
    for got, want in zip(grads, torch.autograd.grad(expected, wide, g.double()), strict=True):
        torch.testing.assert_close(got, want.to(dtype), **TOL[dtype])


def test_under_autocast_csr_names_the_dtypes_and_dense_runs(device, backend):
    low = torch.float16 if device == "cuda" else torch.bfloat16
    model = nn.Sequential(nn.Linear(6, 6), SparseLinear(6, 5, density=0.5, backend=backend)).to(device)
    x = torch.randn(4, 6, device=device)
    with torch.autocast(device, dtype=low):
        if backend == "csr":
            with pytest.raises(ValueError, match=f"input is {low} and the values are torch.float32.*autocast"):
                model(x)
        else:
            hidden = model[0](x)
            torch.testing.assert_close(model(x), oracle(model[1], hidden), **TOL[low])
    assert torch.ones(2, device=device).add(1).sum().item() == 4  # the device still works


MOVES = {
    "double": lambda layer: layer.double(),
    "deepcopy": copy.deepcopy,
    "half": pytest.param(lambda layer: layer.cuda().half(), marks=pytest.mark.cuda),
    "cuda": pytest.param(lambda layer: layer.cuda(), marks=pytest.mark.cuda),
    "cpu": pytest.param(lambda layer: layer.cuda().cpu(), marks=pytest.mark.cuda),
}


@pytest.mark.parametrize("move", MOVES.values(), ids=MOVES.keys())
def test_a_moved_layer_still_matches_the_oracle(any_backend, move):
    layer = move(SparseLinear(12, 7, density=0.4, capacity=40, backend=any_backend))
    values = layer.weight.values
    fmt = layer.weight.format()
    assert all(
        t.dtype == torch.int64 and t.device == values.device for t in [layer.weight.index, *fmt.buffers.values()]
    )
    x = torch.randn(3, 12, device=values.device, dtype=values.dtype)
    torch.testing.assert_close(layer(x), oracle(layer, x), **TOL[values.dtype])


def test_csr_layer_compiles_fullgraph(device, close):
    layer = SparseLinear(12, 7, density=0.4, capacity=40, backend="csr").to(device)
    x = torch.randn(2, 3, 12, device=device, requires_grad=True)
    compiled = torch.compile(layer, fullgraph=True)
    assert layer.bias is not None
    leaves = (layer.weight.values, x, layer.bias)
    for got, want in zip(
        (compiled(x), *torch.autograd.grad(compiled(x).square().sum(), leaves)),
        (layer(x), *torch.autograd.grad(layer(x).square().sum(), leaves)),
        strict=True,
    ):
        close(got, want)


@pytest.mark.cuda
def test_cuda_graph_replays_a_csr_training_step():
    def build():
        torch.manual_seed(0)
        model = nn.Sequential(
            SparseLinear(64, 64, density=0.2, backend="csr"),
            nn.ReLU(),
            SparseLinear(64, 10, density=0.2, backend="csr"),
        ).cuda()
        return model, torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    x = torch.randn(32, 64, device="cuda")
    target = torch.randint(10, (32,), device="cuda")

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), target)
        loss.backward()
        optimizer.step()
        return loss

    eager, eager_optimizer = build()
    expected = [step(eager, eager_optimizer).item() for _ in range(23)]

    model, optimizer = build()
    side = torch.cuda.Stream()  # warm up off the default stream, as capture requires
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = [step(model, optimizer).item() for _ in range(3)]
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        loss = step(model, optimizer)
    losses = []
    for _ in range(20):
        graph.replay()
        losses.append(loss.item())
    torch.testing.assert_close(warm + losses, expected, **TOL[torch.float32])
    assert all(later < earlier for earlier, later in pairwise(losses))  # every replay trains


def test_csr_backward_runs_only_the_products_a_gradient_needs(monkeypatch):
    # The first layer of a network needs no input gradient, so no transposed product; frozen
    # values need no sddmm.
    calls = []
    for name in ("spmm", "sddmm"):
        kernel = getattr(_csr, name)
        monkeypatch.setattr(_csr, name, lambda *a, _name=name, _kernel=kernel: calls.append(_name) or _kernel(*a))
    layer = SparseLinear(6, 5, density=0.5, backend="csr")
    for values_grad, x_grad, expected in ((True, False, ["sddmm"]), (False, True, ["spmm"])):
        layer.weight.values.requires_grad_(values_grad)
        y = layer(torch.randn(2, 3, 6, requires_grad=x_grad))
        calls.clear()  # the forward is one spmm
        y.sum().backward()
        assert calls == expected
