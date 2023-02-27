# SBArchOpt: Surrogate-Based Architecture Optimization

This library provides a set of classes and interfaces for applying Surrogate-Based Optimization (SBO)
for architecture optimization problems:
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
| Hierarchical (HIER)     | Imputation; activeness; low imputation ratio    | Impute after sampling, evaluation          | Impute after sampling, evaluation, during infill search; hierarchical kernels      |
| Hidden constraints (HC) | Catch errors and return NaN                     | Extreme barrier approach                   | Predict hidden constraints area                                                    |
| Expensive (EXP)         |                                                 | Use SBO to reduce function evaluations     | Intermediary results storage; resuming optimizations                               |

For MOEA's all measure are already implemented for most algorithms (incl NSGA2).
Only care should be taken to select a repaired sampler so that the initial population is sampled correctly.

SBO measure implementation status
(Lib = yes, in the library; SBArchOpt = yes, in SBArchOpt; N = not implemented; empty = unknown):

| Aspect: measure                       | SBArchOpt SBO | SEGOMOE | pysamoo | BoTorch | Trieste |
|---------------------------------------|---------------|---------|---------|---------|---------|
| MD: continuous relaxation             | SBArchOpt     | Lib     |         |         |         |
| MD: kernels                           | N             | Lib     |         |         |         |
| MD: dummy coding                      | N             | Lib     |         |         |         |
| MD: force new infill point selection  | SBArchOpt     | N       |         |         |         |
| MO: multi-objective infill            | SBArchOpt     | Lib     |         |         |         |
| HIER: imputation during sampling      | SBArchOpt     | N       |         |         |         |
| HIER: imputation after evaluation     | SBArchOpt     | N       |         |         |         |
| HIER: imputation during infill search | SBArchOpt     | N       |         |         |         |
| HIER: kernels                         | N             | N       | N       | N       | N       |
| HC: predict area                      | N             | N       |         |         | Lib     |
| EXP: intermediary result storage      | N             | N       |         |         |         |
| EXP: resuming optimizations           | N             | N       |         |         |         |

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
