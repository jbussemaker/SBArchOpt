# SBArchOpt: Surrogate-Based Architecture Optimization

SBArchOpt (es-bee-ARK-opt) provides a set of classes and interfaces for applying Surrogate-Based Optimization (SBO)
for system architecture optimization problems:
- Expensive black-box problems: evaluating one candidate architecture might computationally expensive
- Mixed-discrete design variables: categorical architectural decisions mixed with continuous sizing variables
- Hierarchical design variables: decisions can deactivate/activate (parts of) downstream decisions
- Multiple conflicting objectives: stemming from conflicting stakeholder needs
- Subject to hidden constraints: simulation tools might not converge for all design points

Surrogate-Based Optimization (SBO) aims to accelerate convergence by fitting a surrogate model
(e.g. regression, gaussian process, neural net) to the inputs (design variables) and outputs (objectives/constraints)
to try to predict where interesting infill points lie. Potentially, SBO needs about one or two orders of magnitude less
function evaluations than Multi-Objective Evolutionary Algorithms (MOEA's) like NSGA2. However, dealing with the
specific challenges of architecture optimization, especially in a combination of the challenges, is not trivial.
This library hopes to support in doing this.

The library provides:
- A common interface for defining architecture optimization problems based on [pymoo](https://pymoo.org/)
- Support in using Surrogate-Based Optimization (SBO) algorithms:
  - Implementation of a basic SBO algorithm
  - Connectors to various external SBO libraries
- Analytical and realistic test problems that exhibit one or more of the architecture optimization challenges

## Architecture Optimization Measures

To increase the efficiency (and in some cases make it possible at all) of architecture optimization problems, several
measures have been identified. Each of these measures can be implemented independently, however the more, the better.
Architecture optimization aspects and mitigation measures:

| Aspect                  | Problem-level                                   | MOEA                                       | SBO                                                                                |
|-------------------------|-------------------------------------------------|--------------------------------------------|------------------------------------------------------------------------------------|
| Mixed-discrete (MD)     | Convert float to int; high distance correlation | Support discrete operations                | Cont. relaxation; specific kernels; dummy coding; force new infill point selection |
| Multi-objective (MO)    |                                                 | Prioritize w.r.t. distance to Pareto front | Multi-objective infill criteria                                                    |
| Hierarchical (HIER)     | Imputation; activeness; low imputation ratio    | Impute during sampling, evaluation         | Impute during sampling, evaluation, infill search; hierarchical kernels            |
| Hidden constraints (HC) | Catch errors and return NaN                     | Extreme barrier approach                   | Predict hidden constraints area                                                    |
| Expensive (EXP)         |                                                 | Use SBO to reduce function evaluations     | Intermediate results storage; resuming optimizations                               |

Architecture optimization measure implementation status
(Lib = yes, in the library; SBArchOpt = yes, in SBArchOpt; N = not implemented; empty = unknown or not relevant):

| Aspect: measure                       | pymoo     | SBArchOpt SBO | SEGOMOE | pysamoo | BoTorch | Trieste |
|---------------------------------------|-----------|---------------|---------|---------|---------|---------|
| MD: continuous relaxation             |           | SBArchOpt     | Lib     |         |         |         |
| MD: kernels                           |           | N             | Lib     |         |         |         |
| MD: dummy coding                      |           | N             | Lib     |         |         |         |
| MD: force new infill point selection  |           | SBArchOpt     | N       |         |         |         |
| MO: multi-objective infill            |           | SBArchOpt     | Lib     |         |         |         |
| HIER: imputation during sampling      | SBArchOpt | SBArchOpt     | N       |         |         |         |
| HIER: imputation during evaluation    | SBArchOpt | SBArchOpt     | N       |         |         |         |
| HIER: imputation during infill search |           | SBArchOpt     | N       |         |         |         |
| HIER: kernels                         |           | N             | N       | N       | N       | N       |
| HC: predict area                      |           | N             | N       |         |         | Lib     |
| EXP: intermediate result storage      | SBArchOpt | SBArchOpt     | N       |         |         |         |
| EXP: resuming optimizations           | SBArchOpt | SBArchOpt     | N       |         |         |         |

## Installation

First, create a conda environment (skip if you already have one):
```
conda create --name opt python=3.9
conda activate opt
```

Then install the package:
```
conda install numpy
python setup.py install
```

## Usage

### Connecting to Optimization Frameworks

Interfaces to optimization frameworks are located in the `sb_arch_opt.algo` module.
Each framework has a dedicated readme file with instructions on how to use and how to install the framework.

Optimization frameworks (except pymoo) are always optional! Therefore, when installing SBArchOpt, the interfaces to
optimization frameworks are probably not working yet.

### Test Problems

For an overview of the available test problems and how to use them, see the `readme_problems.md` in
`sb_arch_opt.problems`.

### Implementing an Architecture Optimization Problem

To implement an architecture optimization problem, create a new class extending the `ArchOptProblemBase` class.
You then need to implement the following functionality:
- Design variable definition in the `__init__` function using `Variable` classes (in `pymoo.core.variable`)
- Evaluation of a design vector in `_arch_evaluate`
- Correction (imputation/repair) of a design vector in `_correct_x`
- An interface for implementing intermediate storage of problem-specific results (`store_results`), and restarting an
  optimization from these previous results (`load_previous_results`)
- A unique class representation in `__repr__`

Design variables of different types are defined as follows:
- Continuous (`Real`): any value between some lower and upper bound (inclusive)
  --> for example [0, 1]: 0, 0.25, .667, 1
- Integer (`Integer` or `Binary`): integer between 0 and some upper bound (inclusive); ordering and distance matters
  --> for example [0, 2]: 0, 1, 2
- Categorical (`Choice`): one of n options, encoded as integers; ordering and distance are not defined
  --> for example [red, blue, yellow]: red, blue (to get associated categorical values, use `get_categorical_values`)

The input to the `_arch_evaluate` function has not yet been imputed (however discrete variables have been rounded and
therefore have integer values). Output of the evaluation should be provided as follows:
- Objective function `f` as minimization
- Inequality constraints `g` according to `g(x) <= 0` means "satisfied"
- Equality constraints `h` according to `h(x) = 0` means "satisfied"

Note: if you are implementing a test problem where it is relatively cheap to determine the "real" Pareto front, you
may also extend the `sb_arch_opt.pareto_front.ArchOptTestProblemBase` class. This class has the same features as the
other base class, however enables the use of the `pareto_front()` function to get a reference Pareto front for testing.

Example:

```python
import numpy as np
from sb_arch_opt.problem import ArchOptProblemBase
from pymoo.core.variable import Real, Integer, Choice


class DemoArchOptProblem(ArchOptProblemBase):

    def __init__(self):
        super().__init__(des_vars=[
            Real(bounds=(0, 1)),
            Integer(bounds=(0, 3)),  # [0, 1, 2, 3]
            Choice(options=['A', 'B', 'C']),
        ], n_obj=1)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """
        Implement evaluation and write results in the provided output matrices:
        - x (design vectors): discrete variables have integer values, imputed design vectors can be output here
        - is_active (activeness): vector specifying for each design variable whether it was active or not
        - f (objectives): written as a minimization
        - g (inequality constraints): written as "<= 0"
        - h (equality constraints): written as "= 0"
        """

        # Correct the input design vectors (if not too expensive)
        self._correct_x(x, is_active_out)

        # Example of how to set objective values
        f_out[:, 0] = np.sum(x ** 2, axis=1)

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """Impute the design vectors (discrete variables already have integer values),
        writing the imputed design vectors and the activeness matrix to the provided matrices"""

        # Get categorical values associated to the third design variables (i_dv = 2)
        categorical_values = self.get_categorical_values(x, i_dv=2)

        # Set second design variable inactive if value is other than A
        is_active[:, 1] = categorical_values != 'A'
        x[~is_active] = 0

    def store_results(self, results_folder, final=False):
        """Implement this function to enable problem-specific intermediate results storage"""

    def load_previous_results(self, results_folder):
        """Implement this function to enable problem-specific results loading for algorithm restart"""

    def __repr__(self):
        """repr() of the class, should be unique for unique Pareto fronts"""
        return f'{self.__class__.__name__}()'
```
