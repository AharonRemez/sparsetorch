import pytest
import torch

from spartorch.init import fan_in_uniform_


def test_each_row_within_its_own_bound(device):
    # Row 0 has 4 connections, row 1 one, row 2 none; slots 5 and 6 are free.
    index = torch.tensor([0, 1, 2, 3, 9, -1, -1], device=device)
    values = torch.full((7,), 7.0, device=device)
    lone = []
    for _ in range(50):
        assert fan_in_uniform_(values, index, (3, 8)) is values
        assert (values[:4].abs() <= 0.5).all()
        assert torch.equal(values[5:], torch.full((2,), 7.0, device=device))
        lone.append(values[4].abs().item())
    assert 0.5 < max(lone) <= 1  # 50 draws from U(-1, 1) all stay inside 0.5 with probability 2^-50


DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp64", "fp32", "fp16", "bf16"])
def test_row_variance_is_one_over_three_fan_in(dtype):
    shape = (50, 2000)
    index = torch.randperm(100_000)[:50_000]
    values = fan_in_uniform_(torch.empty(50_000, dtype=dtype), index, shape).double()
    row = index // shape[1]
    k = torch.bincount(row, minlength=50)
    variance = torch.zeros(50, dtype=torch.float64).index_add_(0, row, values.square()) / k
    # k is about 1000 per row, so the variance estimate has a relative std of sqrt(0.8 / k) = 2.8 %;
    # 15 % is more than 5 sigma for each of the 50 rows.
    torch.testing.assert_close(variance * 3 * k, torch.ones(50, dtype=torch.float64), rtol=0.15, atol=0)


def test_no_gradient_is_recorded():
    values = torch.zeros(3, requires_grad=True)
    fan_in_uniform_(values, torch.tensor([0, 1, 2]), (1, 3))
    assert values.grad_fn is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_a_row_with_more_connections_than_half_precision_counts(dtype):
    # 70,000 is past fp16's largest finite number, 65,504: as an fp16 count it is inf, and rsqrt(inf) is 0.
    values = fan_in_uniform_(torch.full((70_000,), 7.0, dtype=dtype), torch.arange(70_000), (1, 70_000)).float()
    bound = 70_000**-0.5
    assert (values != 0).float().mean() > 0.99  # an fp16 draw is exactly 0 about once in 2,000
    assert 0.9 * bound < values.abs().max() <= 1.01 * bound  # 1 % covers the rounding of a half-precision bound
