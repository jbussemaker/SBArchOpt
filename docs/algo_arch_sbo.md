# Architecture Surrogate-Based Optimization (SBO) Algorithm

`arch_sbo` implements a Surrogate-Based Optimization (SBO) algorithm configured for solving most types of architecture
optimization problems. It has been developed with experience from the following work:

J.H. Bussemaker et al., "Effectiveness of Surrogate-Based Optimization Algorithms for System Architecture Optimization",
AIAA Aviation 2021, DOI: [10.2514/6.2021-3095](https://arc.aiaa.org/doi/10.2514/6.2021-3095)

The algorithm uses state-of-the-art mixed-discrete hierarchical Gaussian Process (Kriging) surrogate models and ensures
that all evaluated and selected infill points are imputed (and therefore valid).

Either an RBF or a Kriging surrogate model is used. Kriging has as advantage that it also can predict the amount of
uncertainty next to the model mean, and therefore allows for more interesting infill criteria. RBF, however, is faster
to train and evaluate.

## Installation

```
pip install -e .[arch_sbo]
```

## Usage

The algorithm is implemented as a [pymoo](https://pymoo.org/) algorithm and already includes all relevant architecture
optimization measures. It can be used directly with pymoo's interface:

```python
from pymoo.optimize import minimize
from sb_arch_opt.algo.arch_sbo import get_arch_sbo_gp, get_arch_sbo_rbf

problem = ...  # Subclass of ArchOptProblemBase

# Get Kriging or RBF algorithm
n_init = 100
results_folder_path = 'path/to/results/folder'
gp_arch_sbo_algo = get_arch_sbo_gp(problem, init_size=n_init, results_folder=results_folder_path)

# Start from previous results (skipped if no previous results are available)
gp_arch_sbo_algo.initialize_from_previous_results(problem, results_folder_path)

n_infill = 10
result = minimize(problem, gp_arch_sbo_algo, termination=('n_eval', n_init + n_infill))
```
