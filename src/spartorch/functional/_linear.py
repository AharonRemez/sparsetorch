"""The one linear function of spartorch."""

from torch import Tensor

from spartorch.types import Format


def linear(input: Tensor, values: Tensor, format: Format, bias: Tensor | None = None) -> Tensor:
    r""":math:`y = \mathrm{input}\,W^\top + \mathrm{bias}` for the sparse ``W`` stored as ``(values, format)``.

    ``W`` is ``format.shape = (out, in)``: the connection in slot ``s`` sits at row-major position
    ``index[s]`` with weight ``values[s]``, and every other entry is 0. The arguments are
    ``F.linear``'s, with ``(values, format)`` where it has ``weight``. The sizes are checked
    against the format before the kernel of ``format.backend`` runs, from shapes alone, so
    nothing is read back from the device.

    Args:
        input: ``(..., in)``, any number of leading dimensions.
        values: ``(capacity,)``, the weights of the slots.
        format: what :func:`~spartorch.functional.prepare` built from the pattern.
        bias: ``(out,)`` or None.

    Returns:
        ``(..., out)``, differentiable with respect to ``input``, ``values`` and ``bias``.

    Raises:
        ValueError: if ``values`` is not ``(capacity,)``, ``input.shape[-1]`` is not ``in``, or
            ``bias`` is not ``(out,)``, or if the backend has no :attr:`~spartorch.types.Backend.linear` kernel.
    """
    out, in_ = format.shape
    if values.shape != (format.capacity,):
        raise ValueError(f"values has shape {tuple(values.shape)}, the format has capacity {format.capacity}")
    if input.shape[-1] != in_:
        raise ValueError(f"input has {input.shape[-1]} features, the weight has {in_}")
    if bias is not None and bias.shape != (out,):
        raise ValueError(f"bias has shape {tuple(bias.shape)}, the weight has {out} outputs")
    if format.backend.linear is None:
        raise ValueError(f"backend {format.backend.name!r} has no linear")
    return format.backend.linear(values, format, input, bias)
