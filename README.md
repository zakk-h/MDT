# tree-again

Interpretable decision tree chains with black box deferral. Includes the
WeightedSeparable and CLARITree backends in one Python package.

## Installation

From a source checkout:

```bash
python -m pip install .
```

From PyPI:

```bash
python -m pip install tree-again
```

Source installations require a C++17 compiler and Python development headers.
Pip installs the build dependencies, including pybind11 and Eigen headers.
Neither a separate CLARITree installation nor CMake is required. Compatible
prebuilt wheels do not require a compiler on the user's machine.

## Usage

```python
from tree_again import MDTChain, CLARITree, WeightedSeparable
```

A small binary classification example:

```python
import numpy as np
from tree_again import MDTChain

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 8, dtype=np.uint8)
y = X[:, 0]

model = MDTChain(
    training_method="top_down",
    c=0.5,
    defer_lambda_strength=1.0,
    lambda_leaf=0.001,
    eta_defer=1.0,
    depth_budget=2,
    max_stages=1,
    task="classification",
    random_state=0,
)
model.fit(X, y)
predictions = model.predict(X)
```

To use CLARITree stages, select `task="linear_regression"` and supply backend
options through `claritree_params`. CLARITree is included in the same wheel.
Its direct classes are also available from `tree_again.clari_tree`.

TabPFN is optional:

```bash
python -m pip install 'tree-again[tabpfn]'
```

Experiment datasets and stored results are not part of this source release.
The files in `examples/` are preserved experiment scripts, not standalone demos.
