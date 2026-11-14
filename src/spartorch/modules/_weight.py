"""The storage of a sparse weight: its values, its pattern and the format of its backend."""

from collections.abc import Mapping
from math import prod
from typing import Any

import torch
from torch import Tensor, nn

from spartorch import functional
from spartorch.types import Backend, Format


class SparseWeight(nn.Module):
    """A sparse weight of dense ``shape`` in ``capacity`` slots.

    ``index[s]`` is the row-major flat position in ``shape`` of the connection in slot ``s``, or
    -1 if the slot is free; ``values[s]`` is its weight. Slots are in no order. ``shape`` is a
    tuple of ints, the one copy of the shape. The ``state_dict`` is ``values``, ``index`` and
    ``shape``: ``shape`` is written from the tuple on save and compared with it on load, where a
    checkpoint of another shape is an error and loads nothing. The buffers of the format
    (:func:`~spartorch.functional.prepare` on ``shape`` and ``index``) are registered so that they follow the
    module's device, are not saved, and are rebuilt from ``index`` after every load.

    Args:
        shape: the dense shape, outputs first.
        index: ``(n,)`` int64 positions, -1 for a free slot.
        capacity: number of slots, at least ``n``; ``n`` if None. Slots past ``n`` start free.
        backend: a preset name of :data:`~spartorch.functional.backends.PRESETS` or a
            :class:`~spartorch.types.Backend`; ``"csr"`` by default.

    Raises:
        ValueError: for a pattern :func:`~spartorch.functional.sort_pattern` rejects, a ``capacity`` below ``n``, or an
            unknown backend.
    """

    index: Tensor

    def __init__(
        self, shape: tuple[int, ...], index: Tensor, capacity: int | None = None, *, backend: str | Backend = "csr"
    ):
        super().__init__()
        self.shape = tuple(shape)
        # Checked before the free slots are appended, which a 2-D index would fail with a RuntimeError.
        functional.sort_pattern(index, prod(self.shape))
        n = index.numel()
        capacity = n if capacity is None else capacity
        if capacity < n:
            raise ValueError(f"capacity {capacity} is below the {n} connections")
        self.backend = functional.backends.resolve(backend)
        self.values = nn.Parameter(torch.zeros(capacity, device=index.device))
        self.register_buffer("index", torch.cat([index, index.new_full((capacity - n,), -1)]))
        self._prepare()

    def _prepare(self) -> None:
        buffers = functional.prepare(self.shape, self.index, backend=self.backend).buffers
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=False)
        self._derived = tuple(buffers)

    def format(self) -> Format:
        """The :class:`~spartorch.types.Format` of the pattern, on the module's device.

        What :func:`~spartorch.functional.linear` takes.
        """
        return Format(
            self.backend, self.shape, self.index.numel(), {name: getattr(self, name) for name in self._derived}
        )

    def _save_to_state_dict(self, destination: dict[str, Any], prefix: str, keep_vars: bool) -> None:  # noqa: FBT001 -- torch's signature
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "shape"] = torch.tensor(self.shape)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,  # noqa: FBT001 -- torch's signature
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = prefix + "shape"
        if key not in state_dict:
            missing_keys.append(key)
        elif (saved := tuple(state_dict[key].tolist())) != self.shape:
            error_msgs.append(
                f"shape mismatch for {key}: the checkpoint has shape {saved}, this weight has shape {self.shape}."
            )
            return
        rest = {k: v for k, v in state_dict.items() if k != key}
        super()._load_from_state_dict(rest, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._prepare()
