# Tree-structured Parzen Estimator (TPE) Algorithm

A TPE inverts the typical prediction process of a surrogate model: it models x for given y. This allows it to model
very complicated design spaces structures, making it appropriate for architecture optimization and hyperparameter
optimization where it was first developed. For more details, refer to:

Bergstra et al., "Algorithms for Hyper-Parameter Optimization", 2011, available
[here](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf).

We use the implementation found [here](https://github.com/nabenabe0928/tpe), which currently supports single-objective
unconstrained optimization problems.

## Installation

```
pip install sb-arch-opt[tpe]
```

## Usage

The algorithm is implemented as a [pymoo](https://pymoo.org/) algorithm that already includes relevant architecture
optimization measures. It can be used directly with pymoo's interface:

```python
from pymoo.optimize import minimize
from sb_arch_opt.algo.tpe_interface import *

problem = ...  # Subclass of ArchOptProblemBase

# Enable intermediate results storage
results_folder_path = 'path/to/results/folder'

# Get TPE algorithm
n_init = 100
algo = TPEAlgorithm(n_init=n_init, results_folder=results_folder_path)

# Start from previous results (skipped if no previous results are available)
if initialize_from_previous_results(algo, problem, results_folder_path):
    # No need to evaluate any initial points, as they already have been evaluated
    n_init = 0

n_infill = 10
result = minimize(problem, algo, termination=('n_eval', n_init + n_infill))
```
