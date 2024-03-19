"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from sb_arch_opt.problem import *
from sb_arch_opt.algo.arch_sbo.models import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.arch_sbo.metrics import *
from sb_arch_opt.algo.arch_sbo.hc_strategy import *

if not HAS_ARCH_SBO:
    def get_sbo_termination(*_, **__):
        return None


__all__ = ['get_arch_sbo_rbf', 'get_arch_sbo_gp', 'HAS_ARCH_SBO', 'HAS_SMT', 'get_sbo_termination', 'get_sbo',
           'ConstraintAggregation']


def get_arch_sbo_rbf(init_size: int = 100, results_folder=None, **kwargs) -> InfillAlgorithm:
    """
    Get a architecture SBO algorithm using an RBF model as its surrogate model.
    """
    model = ModelFactory.get_rbf_model()
    hc_strategy = get_hc_strategy()
    return get_sbo(model, FunctionEstimateInfill(), init_size=init_size, results_folder=results_folder,
                   hc_strategy=hc_strategy, **kwargs)


def get_arch_sbo_gp(problem: ArchOptProblemBase, init_size: int = 100, n_parallel=None, min_pof: float = None,
                    kpls_n_dim: int = 10, g_aggregation: ConstraintAggregation = None, results_folder=None, **kwargs) \
        -> InfillAlgorithm:
    """
    Get an architecture SBO algorithm using a mixed-discrete Gaussian Process (Kriging) model as its surrogate model.
    Appropriate (multi-objective) infills and constraint handling techniques are automatically selected.

    For constraint handling, increase min_pof to between 0.50 and 0.75 to be more conservative (i.e. require a higher
    probability of being feasible for infill points) or decrease below 0.50 to be more exploratory. Optionally defined
    an aggregation strategy to reduce the number of models to train.

    To reduce model training times for high-dimensional problems, KPLS is used instead of Kriging when the problem
    dimension exceeds kpls_n_dim. Note that the DoE should then contain at least kpls_n_dim+1 points.
    """

    # Create the mixed-discrete Kriging model, correctly configured for the given design space
    kpls_n_comp = kpls_n_dim if kpls_n_dim is not None and problem.n_var > kpls_n_dim else None
    model, normalization = ModelFactory(problem).get_md_kriging_model(kpls_n_comp=kpls_n_comp)

    # Select the single- or multi-objective infill criterion, including constraint handling strategy
    infill, infill_batch = get_default_infill(
        problem, n_parallel=n_parallel, min_pof=min_pof, g_aggregation=g_aggregation)

    # Get default hidden constraint strategy
    hc_strategy = get_hc_strategy(kpls_n_dim=kpls_n_dim)

    return get_sbo(model, infill, infill_size=infill_batch, init_size=init_size, normalization=normalization,
                   results_folder=results_folder, hc_strategy=hc_strategy, **kwargs)


def get_sbo(surrogate_model, infill: 'SurrogateInfill', infill_size: int = 1, init_size: int = 100,
            infill_pop_size: int = 100, infill_gens: int = None, repair=None, normalization=None,
            hc_strategy: 'HiddenConstraintStrategy' = None, results_folder=None, **kwargs) -> InfillAlgorithm:
    """Create the SBO algorithm given some SMT surrogate model and an infill criterion"""

    sbo = SBOInfill(surrogate_model, infill, pop_size=infill_pop_size, termination=infill_gens, repair=repair,
                    normalization=normalization, hc_strategy=hc_strategy, verbose=True)\
        .algorithm(infill_size=infill_size, init_size=init_size, **kwargs)

    if results_folder is not None:
        sbo.store_intermediate_results(results_folder=results_folder)
    return sbo
