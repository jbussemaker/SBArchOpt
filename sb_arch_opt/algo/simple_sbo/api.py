"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
from sb_arch_opt.problem import ArchOptRepair
from sb_arch_opt.algo.simple_sbo.models import *
from sb_arch_opt.algo.simple_sbo.algo import *
from sb_arch_opt.algo.simple_sbo.infill import *
from sb_arch_opt.algo.simple_sbo.metrics import *

if not HAS_SIMPLE_SBO:
    get_sbo_termination = lambda *_, **__: None


__all__ = ['get_simple_sbo_rbf', 'get_simple_sbo_krg', 'HAS_SIMPLE_SBO', 'get_sbo_termination']


def get_simple_sbo_rbf(init_size: int = 100, **kwargs):
    """
    Get a simple SBO algorithm using an RBF model as its surrogate model.
    """
    check_dependencies()
    model = ModelFactory.get_rbf_model()
    return get_sbo(model, FunctionEstimateInfill(), init_size=init_size, **kwargs)


def get_simple_sbo_krg(init_size: int = 100, use_mvpf=True, use_ei=False, min_pof=None, **kwargs):
    """
    Get a simple SBO algorithm using a Kriging model as its surrogate model.
    It can use one of the following infill strategies:
    - Expected improvement (multi-objectified)
    - Minimum Variance of the Pareto Front (MVPF)
    - Directly optimizing on the mean prediction
    All strategies support constraints.
    """
    check_dependencies()
    model = ModelFactory.get_kriging_model()
    if use_ei:
        infill = ExpectedImprovementInfill(min_pof=min_pof)  # For single objective
    else:
        infill = MinVariancePFInfill(min_pof=min_pof) if use_mvpf else FunctionEstimateConstrainedInfill(min_pof=min_pof)
    return get_sbo(model, infill, init_size=init_size, **kwargs)


def get_sbo(surrogate_model, infill: 'SurrogateInfill', infill_size: int = 1, init_size: int = 100,
            infill_pop_size: int = 100, infill_gens: int = 100, repair=None, **kwargs):
    """Create the SBO algorithm given some SMT surrogate model and an infill criterion"""
    if repair is None:
        repair = ArchOptRepair()

    return SBOInfill(surrogate_model, infill, pop_size=infill_pop_size, termination=infill_gens, repair=repair,
                     verbose=True).algorithm(infill_size=infill_size, init_size=init_size, **kwargs)
