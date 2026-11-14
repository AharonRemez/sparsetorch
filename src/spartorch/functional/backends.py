"""The presets (a storage format paired with its kernels), where a name is resolved, and where a format is made."""

from math import prod

from torch import Tensor

from spartorch.functional._topology import sort_pattern
from spartorch.functional.ops import linear_csr_format, linear_dense
from spartorch.functional.storage import prepare_csr, prepare_dense
from spartorch.types import Backend, Format

PRESETS: dict[str, Backend] = {
    "csr": Backend("csr", prepare=prepare_csr, linear=linear_csr_format),
    "dense": Backend("dense", prepare=prepare_dense, linear=linear_dense),
}
"""The backends known by name.

``"csr"``, the default: :func:`~spartorch.functional.storage.prepare_csr` with
:func:`~spartorch.functional.ops.linear_csr_format`, truly sparse products through ``torch.sparse``
(cuSPARSE on CUDA); the input must have the dtype of the values, so it does not run under autocast.
``"dense"``: :func:`~spartorch.functional.storage.prepare_dense` with
:func:`~spartorch.functional.ops.linear_dense`, the values scattered into a dense weight and
``F.linear``; it runs in every dtype and under autocast.
"""


def resolve(backend: str | Backend) -> Backend:
    """The :class:`~spartorch.types.Backend` a preset name stands for, or ``backend`` itself if it is one.

    Args:
        backend: a preset name of :data:`~spartorch.functional.backends.PRESETS`, or a
            :class:`~spartorch.types.Backend`.

    Raises:
        ValueError: for a name that is not in :data:`~spartorch.functional.backends.PRESETS`.
    """
    if isinstance(backend, Backend):
        return backend
    if backend not in PRESETS:
        raise ValueError(f"unknown backend {backend!r}; presets: {sorted(PRESETS)}")
    return PRESETS[backend]


def prepare(shape: tuple[int, ...], index: Tensor, *, backend: str | Backend = "csr") -> Format:
    """The :class:`~spartorch.types.Format` of a pattern.

    :func:`~spartorch.functional.sort_pattern`, then the backend's :attr:`~spartorch.types.Backend.prepare`. The one
    way a :class:`~spartorch.types.Format` is made. It reads the pattern and changes nothing.

    Args:
        shape: the dense shape of the weight, outputs first.
        index: ``(capacity,)`` int64 row-major flat positions in ``shape``, -1 for a free slot.
        backend: a preset name of :data:`~spartorch.functional.backends.PRESETS`, or a
            :class:`~spartorch.types.Backend`; ``"csr"`` by default.

    Returns:
        The buffers that ``backend.prepare`` derives, with the resolved backend, ``shape`` and
        ``capacity = index.numel()``.

    Raises:
        ValueError: for a pattern :func:`~spartorch.functional.sort_pattern` rejects, or an unknown preset.
    """
    resolved = resolve(backend)
    buffers = resolved.prepare(shape, index, sort_pattern(index, prod(shape)))
    return Format(resolved, tuple(shape), index.numel(), buffers)
