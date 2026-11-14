spartorch.functional
====================

.. automodule:: spartorch.functional

.. currentmodule:: spartorch.functional

Layers
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   linear


Patterns
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare
   sort_pattern


Storage formats
---------------

.. automodule:: spartorch.functional.storage

.. autosummary::
   :toctree: generated
   :nosignatures:

   prepare_csr
   prepare_dense


Kernels
-------

.. automodule:: spartorch.functional.ops

.. autosummary::
   :toctree: generated
   :nosignatures:

   linear_csr_format
   linear_dense

The registered ops, under the namespace ``spartorch::``:

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: optemplate.rst

   linear_csr
   sddmm
   spmm


Backends
--------

.. automodule:: spartorch.functional.backends

.. autosummary::
   :toctree: generated
   :nosignatures:

   PRESETS
   resolve
