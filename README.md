# SparTorch

Truly sparse layers for PyTorch. A weight is stored as its connections, the non-zero values
and their positions, and it trains on the GPU. There is no dense weight and no mask.

Most sparse-training libraries keep a dense weight and multiply it by a mask, so a network at
5 % density still costs what a dense one costs to train. SparTorch stores and computes only
what exists, in a form PyTorch treats as ordinary: the values are a normal `nn.Parameter`, so
every optimizer, scheduler, `state_dict` and `.to()` works unchanged.

## Status

This is an early version.

**Works today**

- `SparseLinear`: a sparse counterpart of `nn.Linear`, on CPU and CUDA.
- Two backends for the same weight: `csr` (PyTorch's sparse kernels) and `dense`.
- Everything a class does is also a function, in `spartorch.functional`.
- Saving and loading through the ordinary `state_dict`.

**Not here yet**

- Changing the connections during training (dropping and growing), and with it the dynamic
  sparse training algorithms SET and RigL.
- Sparse convolutions.
- Benchmarks against dense layers.

## Install

Python 3.13 or newer, PyTorch 2.15 or newer, and [uv](https://docs.astral.sh/uv/). The project
is set up for CUDA 13.4 (the `cu134` PyTorch wheels); it also runs on CPU.

```bash
git clone https://github.com/AharonRemez/sparsetorch.git
cd sparsetorch
uv sync
```

## Use

A sparse layer goes where an `nn.Linear` would, inside any model:

```python
import torch
import torch.nn.functional as F
from torch import nn

import spartorch as st

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    st.SparseLinear(784, 300, density=0.05),  # 5 % of the 784 x 300 connections
    nn.ReLU(),
    st.SparseLinear(300, 10, nnz=600),  # or an exact number of connections
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # any torch optimizer
x = torch.randn(128, 784, device=device)
y = torch.randint(10, (128,), device=device)

loss = F.cross_entropy(model(x), y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The layer's parameters are the values of its connections and its bias:

```python
layer = model[0]
layer.weight.values.shape  # torch.Size([11760]): one value per connection
layer.weight.index.shape  # torch.Size([11760]): where each one sits in the 300 x 784 weight
layer.weight.shape  # (300, 784)
```

Save and load as usual. A checkpoint from a layer of another shape is refused:

```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", weights_only=True))
```

### Backends

The same stored weight can be computed in two ways. The choice is always yours; nothing
switches it for you.

```python
st.SparseLinear(784, 300, density=0.05, backend="csr")  # the default
st.SparseLinear(784, 300, density=0.05, backend="dense")
```

| Backend | How it computes | Dtypes | Autocast |
|---|---|---|---|
| `csr` | PyTorch's sparse kernels on the connections only (cuSPARSE on CUDA) | fp32 and fp64 everywhere; fp16 and bf16 on CUDA | not supported: it raises |
| `dense` | scatters the values into a dense weight, then `F.linear` | every floating dtype | supported |

On CUDA the `csr` forward is not bitwise reproducible between runs: results differ by about
one unit in the last place. Its weight gradient is bitwise reproducible. The measured figures
are in the kernels' docstrings.

### The same as functions

`spartorch.functional` is to the classes what `torch.nn.functional` is to `torch.nn`:

```python
import spartorch.functional as SF

index = layer.weight.index  # int64 positions in the flattened weight; -1 marks a free slot
fmt = SF.prepare((300, 784), index, backend="csr")  # checks the pattern, builds the format
out = SF.linear(x, layer.weight.values, fmt, layer.bias)  # the argument order of F.linear
assert torch.allclose(out, layer(x), atol=1e-5)  # the same computation as the layer's forward
```

A backend is a storage format paired with the kernel that reads it. Writing your own is two
functions, and the layer runs on it unchanged:

```python
def my_kernel(values, format, x, bias):
    return SF.ops.linear_dense(values, format, x, bias) * 1.0  # your computation here


mine = st.types.Backend("mine", prepare=SF.storage.prepare_dense, linear=my_kernel)
custom = st.SparseLinear(784, 300, density=0.05, backend=mine).to(device)
custom(x).shape  # torch.Size([128, 300])
```

## How a weight is stored

A `SparseWeight` of shape `(out, in)` has `capacity` slots:

- `values`: an `nn.Parameter` of shape `(capacity,)`, the weight of each slot.
- `index`: an int64 buffer of shape `(capacity,)`, the position of each slot's connection in
  the flattened dense weight (`row * in + col`), or `-1` for a free slot.
- `shape`: the dense shape, saved in the `state_dict` next to the two above.

Slots are in no order and never move, so an optimizer's per-weight state stays aligned with
`values`. What a kernel needs (CSR arrays, for example) is derived from `index` and is not
saved.

## Layout

```
src/spartorch/
  modules/       the classes, as torch.nn: SparseLinear, SparseWeight
  functional/    the functions, as torch.nn.functional
    storage/       storage formats built from a pattern: csr, dense
    ops/           the kernels: registered PyTorch ops
    backends.py    a format paired with its kernel; the presets
  types.py       the records: Backend, Format
  init.py        initialisation, as torch.nn.init
tests/           mirrors src; every result is checked against a dense reference
tools/           mutate.py: mutation testing of the whole package
doc/             the Sphinx documentation
```

## Development

```bash
./ci.sh
```

runs the formatter and linter (ruff), the type checker (pyright), the tests including the CUDA
tests, and builds the documentation with warnings as errors. Enable the pre-push hook once per
clone with `git config core.hooksPath .githooks`.

`uv run python tools/mutate.py` breaks every statement of the package, one at a time, and
checks that a test notices.

## Licence

MIT. See [LICENSE](LICENSE).
