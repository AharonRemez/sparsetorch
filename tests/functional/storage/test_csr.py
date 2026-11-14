import pytest
import torch

from spartorch.functional import sort_pattern
from spartorch.functional.storage import prepare_csr

# Each with free slots, so capacity is above nnz; "rows" has empty rows and columns, the last ones included.
PATTERNS = {
    "rows": ((6, 7), 0.3, [0, 5]),
    "full": ((3, 4), 1.0, []),
    "empty-pattern": ((4, 5), 0.0, []),
    "one-row": ((1, 9), 0.5, []),
    "one-column": ((9, 1), 0.5, []),
    "no-rows": ((0, 5), 0.0, []),
    "no-columns": ((5, 0), 0.0, []),
}


def random_pattern(shape, density, empty_rows, device):
    """A shuffled ``(capacity,)`` index with free slots, and the bool dense matrix it describes."""
    keep = torch.rand(shape, device=device) < density
    keep[empty_rows] = False
    if shape == (6, 7):
        keep[:, [2, 6]] = False
    position = keep.flatten().nonzero().squeeze(1)
    index = torch.cat([position, position.new_full((3,), -1)])
    return index[torch.randperm(index.numel(), device=device)], keep


@pytest.mark.parametrize(("shape", "density", "empty_rows"), PATTERNS.values(), ids=PATTERNS.keys())
def test_the_buffers_are_torchs_csr_of_the_matrix_and_of_its_transpose(device, shape, density, empty_rows):
    index, keep = random_pattern(shape, density, empty_rows, device)
    buffers = prepare_csr(shape, index, sort_pattern(index, keep.numel()))
    # Slot s holds the value s + 1, so a gathered value names the slot it came from.
    slot_values = torch.arange(1, index.numel() + 1, device=device, dtype=torch.float64)
    dense = torch.zeros(keep.numel(), device=device, dtype=torch.float64)
    active = index >= 0
    dense[index[active]] = slot_values[active]
    dense = dense.view(shape)
    gathered = slot_values[buffers["perm"]]
    for (row_ptr, col_idx, values), expected in (
        ((buffers["row_ptr"], buffers["col_idx"], gathered), dense.to_sparse_csr()),
        ((buffers["t_row_ptr"], buffers["t_col_idx"], gathered[buffers["t_perm"]]), dense.t().to_sparse_csr()),
    ):
        assert torch.equal(row_ptr, expected.crow_indices())
        assert torch.equal(col_idx, expected.col_indices())
        assert torch.equal(values, expected.values())
    assert all(t.dtype == torch.int64 and t.device == index.device for t in buffers.values())


def test_the_buffers_hold_nothing_of_the_dense_size():
    buffers = prepare_csr((1000, 2000), torch.tensor([3, -1, 900_000]), torch.tensor([0, 2]))
    assert {name: t.numel() for name, t in buffers.items()} == {
        "perm": 2,
        "row_ptr": 1001,
        "col_idx": 2,
        "t_perm": 2,
        "t_row_ptr": 2001,
        "t_col_idx": 2,
    }
