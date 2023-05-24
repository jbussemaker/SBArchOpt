# SBArchOpt Interface to SEGOMOE: Super Efficient Global Optimization with Mixture of Experts

SEGOMOE is a Bayesian optimization toolbox developed by ONERA and ISAE-SUPAERO. For more information refer to:

Bartoli, N., Lefebvre, T., Dubreuil, S., Olivanti, R., Priem, R., Bons, N., Martins J.R.R.A & Morlier, J. (2019).
Adaptive modeling strategy for constrained global optimization with application to aerodynamic wing design. Aerospace
Science and Technology, (90) 85-102.

Priem, R., Bartoli, N., Diouane, Y., & Sgueglia, A. (2020). Upper trust bound feasibility criterion for mixed
constrained Bayesian optimization with application to aircraft design. Aerospace Science and Technology, 105980.

Saves, P., Bartoli, N., Diouane, Y., Lefebvre, T., Morlier, J., David, C., ... & Defoort, S. (2021, July). Constrained
Bayesian optimization over mixed categorical variables, with application to aircraft design. In Proceedings of the
International Conference on Multidisciplinary Design Optimization of Aerospace Systems (AEROBEST 2021) (pp. 1-758).

## Installation

SEGOMOE is not openly available.

## Usage

SEGOMOE is interacted with through the `SEGOMOEInterface` class. This class has a state containing evaluated (and
failed) points, and requires a directory for results storage. The `run_optimization` function can be used to
run the DOE and infill search.

```python
from sb_arch_opt.algo.segomoe_interface import SEGOMOEInterface

problem = ...  # Subclass of ArchOptProblemBase

# Define folder to store results in
results_folder = ...

# Use Mixture of Experts: automatically identifies clusters in the design space with different best surrogates
# ("experts"). Can be more accurate, however also greatly increases the cost of finding new infill points.
use_moe = True

# Options passed to the Sego class and to model generation, respectively
sego_options = {}
model_options = {}

# Get the interface (automatically initialized if the results folder already contains results)
interface = SEGOMOEInterface(problem, results_folder, n_init=100, n_infill=50, use_moe=use_moe,
                             sego_options=sego_options, model_options=model_options)

# Initialize from other results if you want
interface.initialize_from_previous('path/to/other/results_folder')

# Run the optimization loop incl DOE
interface.run_optimization()

x = interface.x  # (n, nx)
x_failed = interface.x_failed  # (n_failed, nx)
f = interface.f  # (n, nf)
g = interface.g  # (n, ng)
pop = interface.pop  # Population containing all design points
opt = interface.opt  # Population containing optimal points (Pareto front if multi-objective)
```
