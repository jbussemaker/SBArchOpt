![pymoo Logo](https://github.com/anyoptimization/pymoo-data/blob/main/logo.png?raw=true)

# pymoo

[pymoo](https://pymoo.org/) is a multi-objective optimization framework that supports mixed-discrete problem
definitions. It includes many test problems and algorithms, mostly evolutionary algorithms such as a Genetic Algorithm
and the Non-dominated Sorting Genetic Algorithm 2 (NSGA2).

The architecture optimization problem base class is based on pymoo.

## Installation

No further actions required.

## Usage

[API Reference](../api/pymoo.md)

Since the problem definition is based on pymoo, pymoo algorithms work out-of-the-box. However, their effectiveness can
be improved by provisioning them with architecture optimization repair operators and repaired sampling strategies.

```python
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from sb_arch_opt.algo.pymoo_interface import provision_pymoo

problem = ...  # Subclass of ArchOptProblemBase

ga_algorithm = GA(pop_size=100)
provision_pymoo(ga_algorithm)  # See intermediate results storage below
result = minimize(problem, ga_algorithm, termination=('n_gen', 10), seed=42)  # Remove seed when using in production!
```

Or to simply get a ready-to-use NSGA2:
```python
from sb_arch_opt.algo.pymoo_interface import get_nsga2

nsga2 = get_nsga2(pop_size=100)
```

### Intermediate Results Storage and Restarting

Storing intermediate results can be useful in case of a problem during optimization. Similarly, restarting can be useful
for continuing a previously failed optimization, or for adding more generations to a previous optimization run.

To enable intermediate results storage, provide a path to a folder where results can be stored to `provision_pymoo` or
`get_nsga2`.

To restart an optimization from a previous run, intermediate results storage must have been used in that previous run.
To then initialize an algorithm, use the `initialize_from_previous_results` function. Partial results are stored after
each evaluation (or after `problem.get_n_batch_evaluate()` evaluations), so even partially-evaluated populations can
be recovered.

```python
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from sb_arch_opt.algo.pymoo_interface import provision_pymoo, \
    initialize_from_previous_results

problem = ...  # Subclass of ArchOptProblemBase

results_folder_path = 'path/to/results/folder'
ga_algorithm = GA(pop_size=100)

# Enable intermediate results storage
provision_pymoo(ga_algorithm, results_folder=results_folder_path)

# Start from previous results (skipped if no previous results are available)
initialize_from_previous_results(ga_algorithm, problem, results_folder_path)

result = minimize(problem, ga_algorithm, termination=('n_gen', 10))
```

For running large DOE's with intermediate results storage, you can use `get_doe_algo`:

```python
from sb_arch_opt.algo.pymoo_interface import get_doe_algo, \
    load_from_previous_results

problem = ...  # Subclass of ArchOptProblemBase
results_folder_path = 'path/to/results/folder'

# Get DOE algorithm and run
doe_algo = get_doe_algo(doe_size=100, results_folder=results_folder_path)
doe_algo.setup(problem, seed=42)  # Remove seed argument when using in production!
doe_algo.run()

# Evaluate the sampled points
pop = doe_algo.pop

# Load intermediate results in case of crash
pop = load_from_previous_results(problem, results_folder_path)
```
