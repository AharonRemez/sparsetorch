"""The sparse counterpart of ``nn.Linear``."""

import torch
from torch import Tensor, nn

from spartorch import functional
from spartorch.init import fan_in_uniform_
from spartorch.modules._weight import SparseWeight
from spartorch.types import Backend


class SparseLinear(nn.Module):
    r""":math:`y = x W^\top + \mathrm{bias}` with a sparse ``W`` of shape ``(out_features, in_features)``.

    ``W`` is ``self.weight``, a :class:`~spartorch.SparseWeight` whose ``nnz`` connections are drawn uniformly at
    random from all positions; :meth:`~spartorch.SparseLinear.forward` is :func:`~spartorch.functional.linear` on its
    tensors. The parameters are ``weight.values``, ``(capacity,)``, and ``bias``, ``(out_features,)``.

    Args:
        in_features: size of the last input dimension.
        out_features: size of the last output dimension.
        density: fraction of the ``in_features * out_features`` positions that are connected,
            ``nnz = round(density * in_features * out_features)``. Give this or ``nnz``.
        nnz: exact number of connections.
        capacity: number of slots, at least ``nnz``; ``nnz`` if None. The rest start free.
        bias: whether the layer has a bias.
        backend: a preset name of :data:`~spartorch.functional.backends.PRESETS` or a
            :class:`~spartorch.types.Backend`; ``"csr"`` by default.

    Raises:
        ValueError: for both or neither of ``density`` and ``nnz``, an ``nnz`` outside
            ``[0, in_features * out_features]``, a ``capacity`` below ``nnz``, or an unknown backend.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        density: float | None = None,
        nnz: int | None = None,
        capacity: int | None = None,
        bias: bool = True,
        backend: str | Backend = "csr",
    ):
        super().__init__()
        size = in_features * out_features
        if density is not None and nnz is None:
            nnz = round(density * size)
        elif nnz is None or density is not None:
            raise ValueError("give exactly one of density and nnz")
        if not 0 <= nnz <= size:
            raise ValueError(f"nnz {nnz} is outside [0, {size}]")
        # shortcut: randperm materialises every position (8 bytes each); sample without it
        # when layers pass ~1e8 positions.
        self.weight = SparseWeight((out_features, in_features), torch.randperm(size)[:nnz], capacity, backend=backend)
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    @property
    def in_features(self) -> int:
        """Size of the last input dimension, ``weight.shape[1]``."""
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        """Size of the last output dimension, ``weight.shape[0]``."""
        return self.weight.shape[0]

    def reset_parameters(self) -> None:
        r"""Redraw the parameters in place, on their device.

        ``weight.values``: :func:`~spartorch.init.fan_in_uniform_`, row ``i`` from
        :math:`U(-1/\sqrt{k_i}, 1/\sqrt{k_i})` with :math:`k_i` its number of connections. ``bias``: ``nn.Linear``'s
        :math:`U(-1/\sqrt{\mathrm{in\_features}}, 1/\sqrt{\mathrm{in\_features}})`, and 0 when ``in_features`` is 0.
        """
        fan_in_uniform_(self.weight.values, self.weight.index, self.weight.shape)
        if self.bias is not None:
            bound = self.in_features**-0.5 if self.in_features else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """:func:`~spartorch.functional.linear` on the layer's tensors.

        Args:
            input: ``(..., in_features)``.

        Returns:
            ``(..., out_features)``.
        """
        return functional.linear(input, self.weight.values, self.weight.format(), self.bias)
