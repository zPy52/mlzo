---
title: Installation
description: Install the MLZO Python alpha and choose optional dependencies.
---

MLZO requires Python 3.12 or newer and PyTorch `>=2.2.2,<3`. Install this alpha from its source repository. It has not been published to PyPI.

## Create an environment

```sh
git clone https://github.com/zPy52/mlzo.git
cd mlzo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mlzo --json --version
```

The final command reports version `0.1.0a1` in a JSON success envelope. `python -m mlzo` and `mlzo` run the same command-line interface.

On Linux CPU machines, install a CPU PyTorch wheel before MLZO if you want to avoid GPU runtime dependencies:

```sh
python -m pip install 'torch>=2.2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

On macOS, use the normal package index. Intel macOS resolves the legacy PyTorch 2.2.2 wheel and uses NumPy below 2 through a package dependency marker. The declared Python and PyTorch ranges are not a tested compatibility matrix.

## Optional dependencies

Install only the adapters you need. These dependencies load when their operation executes.

| Extra | Installs | Enables |
| --- | --- | --- |
| `vision` | Pillow | Image directory decoding |
| `s3` | boto3 | Reading an explicit S3 dataset object |
| `rl` | Gymnasium | Opening an environment by ID for DQN |

```sh
python -m pip install -e '.[vision,s3,rl]'
```

You can use a local duck-typed DQN environment without Gymnasium. S3 data reads do not enable S3 checkpoints. For remote execution, install MLZO and PyTorch on the worker too, then make the project's dependencies available to its job environment.

## Check the installation

```sh
python -c 'import mlzo, torch; print(mlzo.__version__); print(torch.__version__)'
python -m mlzo --help
```

Continue with the [CPU quickstart](../quickstart/). It uses synthetic data and needs no GPU, cloud credentials, or dataset downloads.
