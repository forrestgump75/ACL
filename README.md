# RLHFsims

This repository contains code for evaluating an Automatic Curriculum Learning algorithm for maximizing the expected returns in a finite horizon undiscounted Markov Decision Process. The environment is an 8x8 gridworld with three coins, a danger cell, a wall, and a goal cell. Everything, from the size of the gridworld, to the location of objects, can be changed in the code. Model free and model based methods are also compared.

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
