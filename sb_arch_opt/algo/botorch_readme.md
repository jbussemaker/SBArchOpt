![BoTorch Logo](https://github.com/pytorch/botorch/raw/main/botorch_logo_lockup.png)

# BoTorch: Bayesian Optimization with PyTorch

[BoTorch](https://botorch.org/) is a Bayesian optimization framework written on top of the [PyTorch](https://pytorch.org/)
machine learning library. More information:

Bandalat, M. et al, "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization", https://arxiv.org/abs/1910.06403

The framework is mostly interacted with through [Ax](https://ax.dev/).

## Installation

```
python setup.py install[botorch]
```

## Usage

The `get_botorch_interface` function can be used to get an interface object that can be used to create an
`OptimizationLoop` instance, with correctly configured search space, optimization configuration, and evaluation
function.

Ax will take care of selecting the best underlying Bayesian (GP) model for the defined optimization problem. Note that
it will always be some Gaussian Process model and therefore can be relatively expensive.

```python
from sb_arch_opt.algo.botorch_interface import get_botorch_interface

problem = ...  # Subclass of ArchOptProblemBase

# Get the interface and optimization loop
interface = get_botorch_interface(problem)
opt_loop = interface.get_optimization_loop(n_init=100, n_infill=50)

# Run the optimization loop until completion
opt_loop.full_run()

# Extract data as a pymoo Population object
pop = interface.get_population(opt_loop)
```
