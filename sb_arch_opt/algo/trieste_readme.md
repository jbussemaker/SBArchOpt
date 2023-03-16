# Trieste

[Trieste](https://secondmind-labs.github.io/trieste/1.0.0/index.html) is a Bayesian optimization library built on
[TensorFlow](https://www.tensorflow.org/), Google's machine learning framework. Trieste is an evolution of spearmint.

Trieste supports constrained, multi-objective, noisy, multi-fidelity optimization subject to hidden constraints (called
failed regions in the documentation).

For more information:
Picheny et al., "Trieste: Efficiently Exploring The Depths of Black-box Functions with TensorFlow", 2023,
[arXiv:2302.08436](https://arxiv.org/abs/2302.08436)

## Installation

```
pip install -e .[trieste]
```

## Usage

The `get_trieste_optimizer` function can be used to get an interface object that can be used to create an
`ArchOptBayesianOptimizer` instance, with correctly configured search space, optimization configuration, evaluation
function, and possibility to deal with and stay away from hidden constraints.

To speed up the infill process if you are sure you won't have hidden constraints, you can let the
`might_have_hidden_constraints` function of your problem class return False.

```python
from sb_arch_opt.algo.trieste_interface import get_trieste_optimizer

problem = ...  # Subclass of ArchOptProblemBase

# Get the interface and optimization loop
optimizer = get_trieste_optimizer(problem, n_init=100, n_infill=50)

# Start from previous results (skipped if no previous results are available)
results_folder_path = 'path/to/results/folder'
optimizer.initialize_from_previous(results_folder_path)

# Run the optimization loop (the results folder is optional)
result = optimizer.run_optimization(results_folder=results_folder_path)

# Extract data as a pymoo Population object
pop = optimizer.to_population(result.datasets)
```
