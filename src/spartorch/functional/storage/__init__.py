"""How a pattern is represented in memory: one file per format."""

from ._csr import prepare_csr
from ._dense import prepare_dense

__all__ = ["prepare_csr", "prepare_dense"]
