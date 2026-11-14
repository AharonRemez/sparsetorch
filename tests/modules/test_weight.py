import pytest
import torch

from spartorch import SparseWeight, functional
from tests._references import dense_linear


def test_capacity_adds_free_slots():
    weight = SparseWeight((3, 4), torch.tensor([5, 0]), capacity=4)
    assert weight.index.tolist() == [5, 0, -1, -1]
    assert weight.values.shape == (4,)
    assert isinstance(weight.values, torch.nn.Parameter)


def test_capacity_below_the_connections():
    with pytest.raises(ValueError, match="capacity 1 is below the 2 connections"):
        SparseWeight((3, 4), torch.tensor([5, 0]), capacity=1)


def test_state_dict_is_values_index_shape(any_backend):
    weight = SparseWeight((3, 4), torch.tensor([5, 0]), backend=any_backend)
    assert list(weight.state_dict()) == ["values", "index", "shape"]
    assert weight.state_dict()["shape"].tolist() == [3, 4]
    assert weight.shape == (3, 4)
    assert "shape" not in dict(weight.named_buffers())


def test_format_is_functional_prepare_on_its_own_tensors(any_backend):
    index = torch.tensor([11, -1, 2])
    weight = SparseWeight((3, 4), index, capacity=5, backend=any_backend)
    fmt, expected = weight.format(), functional.prepare((3, 4), weight.index, backend=any_backend)
    assert (fmt.backend, fmt.shape, fmt.capacity) == (expected.backend, (3, 4), 5)
    assert fmt.buffers.keys() == expected.buffers.keys()
    assert all(torch.equal(fmt.buffers[name], expected.buffers[name]) for name in expected.buffers)


def test_the_default_backend_is_csr():
    assert SparseWeight((3, 4), torch.tensor([5, 0])).backend is functional.backends.resolve("csr")


@pytest.mark.cuda
def test_format_follows_the_device(any_backend):
    weight = SparseWeight((3, 4), torch.tensor([11, -1, 2]), backend=any_backend).to("cuda")
    assert all(t.is_cuda for t in weight.format().buffers.values())


def test_pattern_is_checked():
    with pytest.raises(ValueError, match="twice"):
        SparseWeight((3, 4), torch.tensor([1, 1]))


def test_pattern_is_checked_before_free_slots_are_appended():
    # Unchecked, a 2-D index meets the 1-D free slots in torch.cat, which raises a RuntimeError.
    with pytest.raises(ValueError, match="1-D"):
        SparseWeight((3, 4), torch.tensor([[1, 2]]), capacity=3)


def test_save_and_load_round_trip():
    saved = SparseWeight((3, 4), torch.tensor([5, 0, -1]))
    with torch.no_grad():
        saved.values.copy_(torch.tensor([1.0, 2.0, 3.0]))
    loaded = SparseWeight((3, 4), torch.tensor([1, 2, 3]))
    loaded.load_state_dict(saved.state_dict())
    assert torch.equal(loaded.index, saved.index)
    assert torch.equal(loaded.values, saved.values)
    assert loaded.shape == (3, 4)


def test_a_checkpoint_of_another_shape_is_refused():
    other = SparseWeight((4, 3), torch.tensor([7, 1]))
    weight = SparseWeight((3, 4), torch.tensor([5, 0]))
    before = weight.index.clone(), weight.values.detach().clone()
    with pytest.raises(RuntimeError, match=r"shape mismatch for shape: .* \(4, 3\), this weight has shape \(3, 4\)"):
        weight.load_state_dict(other.state_dict())
    assert torch.equal(weight.index, before[0])
    assert torch.equal(weight.values, before[1])


def test_a_checkpoint_without_shape_is_missing_a_key():
    state = SparseWeight((3, 4), torch.tensor([5, 0])).state_dict()
    del state["shape"]
    with pytest.raises(RuntimeError, match=r'Missing key.*"shape"'):
        SparseWeight((3, 4), torch.tensor([1, 2])).load_state_dict(state)


def test_after_a_load_the_format_is_the_new_patterns(device, any_backend):
    saved = SparseWeight((3, 4), torch.tensor([11, -1, 2, 4])).to(device)
    weight = SparseWeight((3, 4), torch.tensor([0, 1, 5, 6]), backend=any_backend).to(device)
    with torch.no_grad():
        saved.values.normal_()
    weight.load_state_dict(saved.state_dict())
    x = torch.randn(2, 4, device=device)
    y = functional.linear(x, weight.values, weight.format())
    torch.testing.assert_close(y, dense_linear(x, saved.values, saved.index, (3, 4)))


def test_loading_an_invalid_pattern_raises():
    state = SparseWeight((3, 4), torch.tensor([5, 0])).state_dict()
    state["index"] = torch.tensor([5, 5])
    with pytest.raises(ValueError, match="twice"):
        SparseWeight((3, 4), torch.tensor([1, 2])).load_state_dict(state)
