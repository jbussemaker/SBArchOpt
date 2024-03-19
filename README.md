# SBArchOpt: Surrogate-Based Architecture Optimization

[![Tests](https://github.com/jbussemaker/SBArchOpt/workflows/Tests/badge.svg)](https://github.com/jbussemaker/SBArchOpt/actions/workflows/tests.yml?query=workflow%3ATests)
[![PyPI](https://img.shields.io/pypi/v/sb-arch-opt.svg)](https://pypi.org/project/sb-arch-opt)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![JOSS](https://joss.theoj.org/papers/0b2b765c04d31a4cead77140f82ecba0/status.svg)](https://joss.theoj.org/papers/0b2b765c04d31a4cead77140f82ecba0)
[![Documentation Status](https://readthedocs.org/projects/sbarchopt/badge/?version=latest)](https://sbarchopt.readthedocs.io/en/latest/?badge=latest)

[GitHub Repository](https://github.com/jbussemaker/SBArchOpt) |
[Documentation](https://sbarchopt.readthedocs.io/)

SBArchOpt (es-bee-ARK-opt) provides a set of classes and interfaces for applying Surrogate-Based Optimization (SBO)
for system architecture optimization problems:
- Expensive black-box problems: evaluating one candidate architecture might be computationally expensive
- Mixed-discrete design variables: categorical architectural decisions mixed with continuous sizing variables
- Hierarchical design variables: decisions can deactivate/activate (parts of) downstream decisions
- Multi-objective: stemming from conflicting stakeholder needs
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

## Installation

First, create a conda environment (skip if you already have one):
```
conda create --name opt python=3.9
conda activate opt
```

Then install the package:
```
conda install numpy
pip install sb-arch-opt
```

Note: there are optional dependencies for the connected optimization frameworks and test problems.
Refer to their documentation for dedicated installation instructions.

## Documentation

Refer to the [documentation](https://sbarchopt.readthedocs.io/) for more background on SBArchOpt
and how to implement architecture optimization problems.

## Citing

If you use SBArchOpt in your work, please cite it:

Bussemaker, J.H., (2023). SBArchOpt: Surrogate-Based Architecture Optimization. Journal of Open Source Software, 8(89),
5564, DOI: [10.21105/joss.05564](https://doi.org/10.21105/joss.05564)

## Contributing

The project is coordinated by: Jasper Bussemaker (*jasper.bussemaker at dlr.de*)

If you find a bug or have a feature request, please file an issue using the Github issue tracker.
If you require support for using SBArchOpt or want to collaborate, feel free to contact me.

Contributions are appreciated too:
- Fork the repository
- Add your contributions to the fork
  - Update/add documentation
  - Add tests and make sure they pass (tests are run using `pytest`)
- Read and sign the [Contributor License Agreement (CLA)](https://github.com/jbussemaker/SBArchOpt/blob/main/SBArchOpt%20DLR%20Individual%20Contributor%20License%20Agreement.docx)
  , and send it to the project coordinator
- Issue a pull request into the `dev` branch

### Adding Documentation

```
pip install -r requirements-docs.txt
mkdocs serve
```

Refer to [mkdocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/) documentation
for more information.
