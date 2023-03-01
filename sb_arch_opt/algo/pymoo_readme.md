# Interface to: pymoo

[pymoo](https://pymoo.org/) is a multi-objective optimization framework that supports mixed-discrete problem
definitions. It includes many test problems and algorithms, mostly evolutionary algorithms such as a Genetic Algorithm
and the Non-dominated Sorting Genetic Algorithm 2 (NSGA2).

The architecture optimization problem base class is based on pymoo.

## Installation

No further actions required.

## Usage

Since the problem definition is based on pymoo, pymoo algorithms work out-of-the-box. However, their effectiveness can
be improved by provisioning them with architecture optimization repair operators and repaired sampling strategies.

```python
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from sb_arch_opt.algo.pymoo_interface import provision_pymoo

problem = ...  # Subclass of ArchOptProblemBase

ga_algorithm = GA(pop_size=100)
provision_pymoo(ga_algorithm)
result = minimize(problem, ga_algorithm, termination=('n_gen', 10))
```

Or to simply get a ready-to-use NSGA2:
```python
from sb_arch_opt.algo.pymoo_interface import get_nsga2

nsga2 = get_nsga2(pop_size=100)
```
