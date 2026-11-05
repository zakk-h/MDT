# tree-again

Code for **[Multistage Defer Trees for Hybrid Interpretability: If at First You Can't Succeed, Tree Again](https://openreview.net/forum?id=bl5HDT7AO1)** at NeurIPS 2026.

This package supports learning decision trees that defer to a more complicated model and multistage defer trees: a sequence of trees that defer to the next model, optionally with a black box at the end.

## Installation

```bash
python -m pip install .
```

From PyPI:

```bash
python -m pip install tree-again
```

See [example.ipynb](examples/example.ipynb) for classification and regression examples.

## Citation

```bibtex
@inproceedings{
heile2026multistage,
title={Multistage Defer Trees for Hybrid Interpretability: If at First You Can't Succeed, Tree Again},
author={Anonymous},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=bl5HDT7AO1}
}
```
