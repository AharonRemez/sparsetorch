"""The classes a model is built from, as ``torch.nn``: they hold state and call :mod:`~spartorch.functional`."""

from ._linear import SparseLinear
from ._weight import SparseWeight

__all__ = ["SparseLinear", "SparseWeight"]
