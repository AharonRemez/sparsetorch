# Tests spartorch/types.py.
import dataclasses

import pytest

from spartorch.functional.storage import prepare_dense
from spartorch.types import Backend, Format


def test_a_backend_names_no_kernel_by_default():
    backend = Backend("bare", prepare=prepare_dense)
    assert (backend.linear, backend.conv2d) == (None, None)


def test_a_backend_cannot_be_changed():
    backend = Backend("bare", prepare=prepare_dense)
    with pytest.raises(dataclasses.FrozenInstanceError):
        backend.name = "other"  # pyright: ignore[reportAttributeAccessIssue] -- the error is what is tested


def test_a_format_cannot_be_changed():
    fmt = Format(Backend("bare", prepare=prepare_dense), (4, 8), 3, {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        fmt.shape = (8, 8)  # pyright: ignore[reportAttributeAccessIssue] -- the error is what is tested


def test_a_format_is_equal_only_to_itself():
    # Field equality would compare buffers, and a format of the same fields is not the same format.
    backend, buffers = Backend("bare", prepare=prepare_dense), {}
    one, other = Format(backend, (4, 8), 3, buffers), Format(backend, (4, 8), 3, buffers)
    assert one == one
    assert one != other
    assert len({one, other}) == 2
