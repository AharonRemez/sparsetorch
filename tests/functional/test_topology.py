import pytest
import torch

from spartorch.functional import sort_pattern


def test_active_slots_ascending_by_position(device):
    index = torch.tensor([7, -1, 0, 19, -1, 3], device=device)
    assert sort_pattern(index, 20).tolist() == [2, 5, 0, 3]


def test_all_free(device):
    assert sort_pattern(torch.full((3,), -1, device=device), 4).numel() == 0


@pytest.mark.parametrize(
    ("index", "match"),
    [
        pytest.param(torch.tensor([0, 1], dtype=torch.int32), "int64", id="dtype"),
        pytest.param(torch.tensor([0.0, 1.0]), "int64", id="float"),
        pytest.param(torch.tensor([[0, 1]]), "1-D", id="ndim"),
        pytest.param(torch.tensor([0, -2]), r"outside \[0, 20\)", id="low"),
        pytest.param(torch.tensor([0, 20]), r"outside \[0, 20\)", id="high"),
        pytest.param(torch.tensor([5, -1, 5]), "twice", id="duplicate"),
        pytest.param(torch.tensor([1, 5, -1, 5]), "twice", id="duplicate-above-the-lowest"),
    ],
)
def test_rejects(index, match, device):
    with pytest.raises(ValueError, match=match):
        sort_pattern(index.to(device), 20)


@pytest.mark.cuda
def test_cuda_context_survives_rejections():
    for index in ([0, -2], [0, 20], [0, 1 << 40], [5, 5]):
        with pytest.raises(ValueError, match=r"outside|twice"):
            sort_pattern(torch.tensor(index, device="cuda"), 20)
    torch.cuda.synchronize()
    assert (torch.arange(3, device="cuda") * 2).sum().item() == 6
