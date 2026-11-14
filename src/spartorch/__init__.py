"""Sparse layers for dynamic sparse training, in the shape of ``torch.nn``."""

from spartorch import functional, init, types
from spartorch.modules import SparseLinear, SparseWeight

__all__ = ["SparseLinear", "SparseWeight", "functional", "init", "types"]
