![BoTorch Logo](https://hebo.readthedocs.io/en/latest/_static/hebo.png)

# SBArchOpt Interface to HEBO: Heteroscedastic Evolutionary Bayesian Optimization

[HEBO](https://hebo.readthedocs.io/en/) is a Bayesian optimization algorithm developed by Huawei Noah's Ark lab.
It supports mixed-discrete parameter and several types of underlying probabilistic models.

For more information:
Cowen-Rivers, A.I., et al. 2022. HEBO: pushing the limits of sample-efficient hyper-parameter optimisation. Journal of
Artificial Intelligence Research, 74, pp.1269-1349, DOI: [10.1613/jair.1.13643](https://dx.doi.org/10.1613/jair.1.13643)

## Installation

```
pip install sbarchopt[hebo]
```

## Usage

The `get_hebo_optimizer` function can be used to get an interface object for running the optimization.
The `hebo` object also has an ask-tell interface if needed.

```python
from sb_arch_opt.algo.hebo_interface import get_hebo_optimizer

problem = ...  # Subclass of ArchOptProblemBase

# Get the interface and optimization loop
hebo = get_hebo_optimizer(problem, n_init=100)

# Run the optimization loop
hebo.optimize(n_infill=50)

# Extract data as a pymoo Population object
pop = hebo.pop
```
