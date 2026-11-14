"""The records that cross a boundary between the parts of spartorch; they carry no code of their own."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class Backend:
    r"""A storage format paired with the kernels that read it.

    Attributes:
        name: the name of the pairing, used in error messages.
        prepare: ``(shape, index, perm) -> buffers``, a function of :mod:`~spartorch.functional.storage` that
            derives the format from the pattern after every change. ``shape`` is the dense shape
            of the weight, ``index`` the ``(capacity,)`` int64 positions (-1 for a free slot) and
            ``perm`` the active slots ascending by position, as :func:`~spartorch.functional.sort_pattern` returns.
        linear: ``(values, format, x, bias) -> y``, a kernel of :mod:`~spartorch.functional.ops` computing
            :math:`y = x W^\top + \mathrm{bias}` for ``W`` of ``format.shape = (out, in)`` and ``x``
            of shape ``(..., in)``; None if the backend has none. :func:`~spartorch.functional.linear` has checked
            the sizes of its arguments against the format before it calls the kernel.
        conv2d: ``(values, format, x, bias, stride, padding, dilation, groups) -> y``, a kernel of
            :mod:`~spartorch.functional.ops`; None if the backend has none.
    """

    name: str
    prepare: Callable[[tuple[int, ...], Tensor, Tensor], dict[str, Tensor]]
    linear: Callable[..., Tensor] | None = None
    conv2d: Callable[..., Tensor] | None = None


@dataclass(frozen=True, eq=False)
class Format:
    """The buffers a kernel reads, together with what they were built for.

    Made by :func:`~spartorch.functional.prepare`. The library promises that no argument of a public function can end
    the CUDA context, and a format built by hand instead of by :func:`~spartorch.functional.prepare` is outside that
    promise. The four cannot be passed apart, so they cannot disagree: a kernel is never given a shape or a capacity
    that its buffers were not derived for.

    Attributes:
        backend: whose kernels read these buffers.
        shape: the dense shape the pattern was checked against, outputs first.
        capacity: the number of slots, the length ``values`` must have.
        buffers: what ``backend.prepare`` derived from the pattern; nothing of the dense size.
    """

    backend: Backend
    shape: tuple[int, ...]
    capacity: int
    buffers: Mapping[str, Tensor]
