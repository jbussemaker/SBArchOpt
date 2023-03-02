# Simple Surrogate-Based Optimization (SBO) Algorithm

`simple_sbo` implements a Surrogate-Based Optimization (SBO) algorithm with experience from the following work:

J.H. Bussemaker et al., "Effectiveness of Surrogate-Based Optimization Algorithms for System Architecture Optimization",
AIAA Aviation 2021, DOI: [10.2514/6.2021-3095](https://arc.aiaa.org/doi/10.2514/6.2021-3095)

The algorithm treats discrete variables as continuous variables during infill search, however is specifically adapted
for architecture optimization by applying the repair operator during sampling and infill search.

Either an RBF or a Kriging surrogate model is used. Kriging has as advantage that it also can predict the amount of
uncertainty next to the model mean, and therefore allows for more interesting infill criteria. RBF, however, is faster
to train and evaluate.

## Installation

```
python setup.py install[simple_sbo]
```

## Usage

The algorithm is implemented as a [pymoo](https://pymoo.org/) algorithm and already includes all relevant architecture
optimization measures. It can be used directly with pymoo's interface:

```python
from pymoo.optimize import minimize
from sb_arch_opt.algo.simple_sbo import get_simple_sbo_krg, get_simple_sbo_rbf

problem = ...  # Subclass of ArchOptProblemBase

# Get Kriging or RBF algorithm
n_init = 100
simple_krg_sbo_algo = get_simple_sbo_krg(init_size=n_init, use_mvpf=True, use_ei=False, min_pof=.95)
simple_rbf_sbo_algo = get_simple_sbo_rbf(init_size=n_init)

# Enable intermediate results storage
results_folder_path = 'path/to/results/folder'
simple_krg_sbo_algo.store_intermediate_results(results_folder_path)

# Start from previous results (skipped if no previous results are available)
if simple_krg_sbo_algo.initialize_from_previous_results(problem, results_folder_path):
    # No need to evaluate any initial points, as they have been previously evaluated
    n_init = 0

n_infill = 10
result = minimize(problem, simple_krg_sbo_algo, termination=('n_eval', n_init+n_infill))
```
