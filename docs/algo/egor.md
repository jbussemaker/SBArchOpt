# Egor (from egobox library)

Egor is a surrogate based optimizer provided with the [egobox](https://joss.theoj.org/papers/10.21105/joss.04737) library.

Egor supports single-objective optimization subject to inequality constraints (<=0) with mixed discrete variables. 
The algorithm uses gaussian processes to iteratively approximate the function under minimization adaptively. 
It selects relevant points by minimising an infill criterion allowing to balance the exploitation and the exploration
of design space regions where the minimum may be located. 

Egor comes with the following features:
* management of mixed-discrete variables through continuous relaxation (integer, ordinal or categorical)
* selection of an infill criterion: EI, WB2 (default), WB2S
* possible use of a mixture of gaussian processes with different trend types and correlation kernels (default to single ordinary kriging)
* reduction of the problem input dimension using partial least squared regression (aka KPLS, used when pb dim >= 9) 
* multithreaded implementation allowing to use multicore cpus (for surrogates training and internal multistart optimizations)

It has been developed with experience from the following work:

Bouhlel, M. A., Hwang, J. T., Bartoli, N., Lafage, R., Morlier, J., & Martins, J. R. R. A.
(2019). [A python surrogate modeling framework with derivatives](https://doi.org/10.1016/j.advengsoft.2019.03.005). 
Advances in Engineering Software, 102662. 

Bartoli, N., Lefebvre, T., Dubreuil, S., Olivanti, R., Priem, R., Bons, N., Martins, J. R. R. A.,
& Morlier, J. (2019). [Adaptive modeling strategy for constrained global optimization with application to aerodynamic wing design](https://doi.org/10.1016/j.ast.2019.03.041). 
Aerospace Science and Technology, 90, 85–102.

Bouhlel, M. A., Bartoli, N., Otsmane, A., & Morlier, J. (2016). [Improving kriging surrogates of high-dimensional design models by partial least squares dimension reduction](https://doi.org/10.1007/s00158-015-1395-9). Structural and Multidisciplinary Optimization, 53(5), 935–952. 

## Installation

```
pip install sb-arch-opt[egor]
```

## Usage

[API Reference](../api/egor.md)

The `get_egor_optimizer` function can be used to get an interface object that can be used to create an
`Egor` instance, with correctly configured search space, optimization configuration, evaluation
function. 

This function allows to pass additional keyword arguments to the underlying Egor optimizer.
See help(egobox.Egor) for further available options.

```python
from sb_arch_opt.algo.egor_interface import get_egor_optimizer

problem = ...  # Subclass of ArchOptProblemBase

# Get the interface and optimization loop
egor = get_egor_optimizer(problem, n_init=100, seed=42)

# Start from previous results (optional)
results_folder_path = 'path/to/results/folder'
egor.initialize_from_previous(results_folder_path)

# Run the optimization loop (the results folder to store results is optional)
result = egor.minimize(results_folder=results_folder_path)

# Extract result as numpy arrays
print(f"Minimum {result.y_opt} at {result.x_opt}")
print(f"History {result.x_hist} {result.y_hist}")

# Extract data as a pymoo Population object
pop = egor.pop
```
