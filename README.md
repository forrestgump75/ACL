# RLHFsims

Research code and notebooks for reinforcement learning experiments with learned or binary feedback in gridworld-style environments.

## Project Layout

- `1/`: early prototype code and a simulation notebook
- `binary_feedback/`: dynamic and static binary-feedback experiments
- `model_based/`: model-based methods, including `KUCBVI_new.py`
- `model_free/`: model-free variants and experiment notebooks
- `march 2026/`: newer analysis notebooks

## Dependencies

The Python scripts in this repository import:

- `numpy`
- `cvxpy`

Some notebooks may also rely on standard scientific Python tooling such as Jupyter and plotting libraries.

## Notes

This repository contains both source code and exploratory notebooks. Generated outputs such as arrays, pickles, plots, caches, and archives are ignored by default to keep the repo lightweight.
