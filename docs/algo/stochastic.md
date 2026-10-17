# Stochastic Optimization

Architecture evaluations often depend on parameters that are not chosen by the optimizer: manufacturing tolerances,
operating conditions, material properties. Optimizing against their nominal values yields a design that is optimal on
paper, but that might perform poorly in reality. Stochastic optimization
instead optimizes a statistic of the response distribution, and therefore looks for a design that is insensitive to 
the uncertainties present in the system.

SBArchOpt supports this by evaluating each design point for many realizations of the uncertain parameters, and reducing
the resulting response distributions to the single values the optimizer sees.
The problem is defined with a `StochasticArchOptProblem`, a subclass of`ArchOptProblemBase`, so
a stochastic problem can be solved with pymoo, ArchSBO, or any of the connected optimization frameworks. 
Each design points costs additional `n_evaluations` evaluations of the underlying model, 
which makes Surrogate-Based Optimization (SBO) especially attractive to reduce the number of design points .

To facilitate uncertainty quantification, sampling and distribution storage, SBArchOpt uses 
the objects defined in [OpenTURNS](https://openturns.github.io/).

Three things are defined on top of the design variables:

- A `StochasticParameterSpace`: the joint distribution of the uncertain parameters, assumed to be independent. They
  influence the evaluation but are not chosen by the optimizer, and therefore are NOT design variables.
- A `UQMethod` (`MonteCarlo` or `PolynomialChaos`): handles the uncertainty propagation, 
  and turning the responses into statistics. It is given a budget of `n_evaluations` expensive evaluations per design point.
- A `Scalarization` per objective and constraint (`Mean`, `Margin` or `Quantile`): scalarizes the output distribution to 
  a single number that optimizer sees. If not specified, responses default to `Mean`.

## Installation

```
pip install sb-arch-opt[uncertainty]
```

## Usage

[API Reference](../api/robust.md)

A stochastic problem is a subclass of `StochasticArchOptProblem`, which extends `ArchOptProblemBase`. Compared to a
deterministic problem, evaluation is implemented in `_arch_evaluate_sample` instead of `_arch_evaluate`. This method
is called once per realization, receives it in `parameters`, and writes the responses for that
realization. The class automatically handles the loop over the sample realizations, output distribution construction
and the scalarization to `f`, `g` and `h` for the the optimizer.

Since the result is an `ArchOptProblemBase`, the problem is then handed to an algorithm like any other. See the
[Stochastic Optimization Tutorial](../tutorial_stochastic.ipynb) for a worked example comparing the robust Pareto front
found by NSGA2 and by SBO to the deterministic one.

### Implementing a Stochastic Problem

Implement `_arch_evaluate_sample` rather than `_arch_evaluate`; everything else works as described in
[implementing an architecture optimization problem](../index.md). Uncertain parameters are stored in the parameter space,
and their samples are reused for all design vectors in `x`.

```python
import openturns as ot
from pymoo.core.variable import Real
from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
from sb_arch_opt.uncertainty import MonteCarlo, Mean, StochasticParameter, \
    StochasticParameterSpace


class MyStochasticProblem(StochasticArchOptProblem):

    def __init__(self):
        # The uncertain parameters: not design variables
        param_space = StochasticParameterSpace([
            StochasticParameter('u', ot.Normal(1., .05)),
        ])

        super().__init__(
            [Real(bounds=(-2., 2.))],
            param_space=param_space,
            uq_method=MonteCarlo(n_evaluations=1000, seed=42),
            n_obj=1, obj_scalar=[Mean()],
        )

    def _is_conditionally_active(self):
        return [False]

    def _correct_x(self, x, is_active):
        pass

    def _arch_evaluate_sample(self, x, is_active, f_out, g_out, h_out, parameters,
                              *args, **kwargs):
        # Called once per realization; parameters contains one realization of the parameter space
        f_out[:, 0] = (parameters[0] - x[:, 0])**2
```

All design points share the same sample realizations (common random numbers) during the evaluation, 
ensuring the outputs remain directly comparable. For reproducibility during development, 
the UQ method can be run with a fixed seed so that identical realizations are generated across independent runs.

A failed evaluation (NaN) in any realization fails the whole design point: the response has no distribution and the
optimizer sees NaN, which is how SBArchOpt treats hidden constraint violations elsewhere.

### Uncertainty Quantification Methods

Both methods draw samples from the parameter space using Latin Hypercube Sampling (LHS) 
and use the same number of expensive model evaluations per design point. 
They differ in how the statistics are obtained: Monte Carlo computes them directly from the sampled model outputs,
while PCE first fits a polynomial chaos expansion to those outputs and derives the statistics by performing 
Monte Carlo on the trained cheap surrogate model. In general Monte Carlo requires more simulation code executions 
to compute the statistics compared to PCE. More UQ methods can be implemented by creating a new subclass of `UQMethod`. 

- `MonteCarlo`: the statistics are computed from the `n_evaluations` responses directly. It is the simplest method to
   implement, but slowest to converge.
- `PolynomialChaos`: a polynomial chaos expansion of each response as a function of the uncertain parameters is fitted
  from the `n_evaluations` responses, and the statistics are then taken from `n_metamodel_samples` cheap evaluations. 
  Sampling the trained surrogate model is much cheaper than running the real simulation code,
  which makes this method attractive for expensive evaluation codes. The PCE surrogate model is trained
  with the least-squares method that requires at least `P` samples, but it is commonly recommended to provide 
  between `2P` and `3P`, where `P = (n+d)! / (n! d!)` is the number of polynomial terms for `n` parameters and polynomial 
  degree `d` (Blatman, 2009).

### Scalarization

Each objective and constraint is reduced to one value by the selected `Scalarization`.
The resulting `float` is passed to the chosen optimization algorithm.

- `Mean`: the expected value of the response --> `min(E[F(x)])`
- `Margin`: `min(E[F(x)] + k*sigma[F(x)])`, assuming a Gaussian response. The default `k=1.645` is the one-sided 95%
  interval of a normal distribution; use `direction=1` for a response that is maximized, so that the margin is
  subtracted rather than added.
- `Quantile`: `min(F_q(x))`, the `q` quantile of the response. Makes no assumption about the shape of the distribution.

Custom scalarizations are implemented by extending `Scalarization` and implementing `scalarize`, which receives the
whole `StochasticOutput` and can therefore use anything the fitted distribution offers.

### Sampled Responses

After each evaluation the fitted response distributions are available in the pymoo output dictionary as
`StochasticOutput` objects, in `out['f_stochastic']`, `out['g_stochastic']` and `out['h_stochastic']`. A response that
turns out not to depend on the uncertain parameters is a plain value instead, and a response that failed is NaN.

```python
import numpy as np

problem = ...  # Subclass of StochasticArchOptProblem

out = problem.evaluate(np.array([[.5]]), return_as_dictionary=True)
output = out['f_stochastic'][0, 0]

print(output.mean, output.std, output.quantile(.95))
```

For polynomial chaos each output also carries the fitted expansion in `method_results`, from which Sobol sensitivity
indices are available to see which uncertain parameters drive the spread of a response:

```python
import openturns as ot

sobol = ot.FunctionalChaosSobolIndices(output.method_results)
first_order_u0 = sobol.getSobolIndex(0)
```

## References
- Blatman, G. (2009). *Adaptive sparse polynomial chaos expansions for uncertainty
  propagation and sensitivity analysis*. PhD thesis, Université Blaise Pascal,
  Clermont-Ferrand, France.