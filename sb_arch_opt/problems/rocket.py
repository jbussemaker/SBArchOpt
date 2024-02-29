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

This test suite contains a multi-stage launch vehicle test problem, originally published here:
https://github.com/raul7gs/Space_launcher_benchmark_problem
"""
import enum
import logging
import itertools
import numpy as np
from typing import List, Optional, Tuple
from pymoo.core.variable import Integer, Choice, Real
from sb_arch_opt.problems.rocket_eval import *
from sb_arch_opt.problems.hierarchical import HierarchyProblemBase

__all__ = ['HAS_ROCKET', 'RocketArch', 'LCRocketArch', 'SOLCRocketArch', 'RocketObj']

log = logging.getLogger('sb_arch_opt.rocket')


class RocketArch(HierarchyProblemBase):
    """
    Multi-stage rocket design problem developed in:
    Raúl García Sánchez, "Adaptation of an MDO Platform for System Architecture Optimization", MSc Thesis,
    Delft University of Technology, jan 2024.

    Design variables:
    - Nr of stages [1, 2, 3]
    - For each stage: nr of engines [1, 2, 3], engine type [1 of 6 types], stage length [m]
    - Overall length-to-diameter ratio
    - Rocket head shape: [Cone (cone angle), Ellipse (ellipse length ratio), Semi-sphere]

    Objectives (log10): cost (minimize), payload mass (maximize)
    Constraints: structural, payload volume
    """

    _engines = [Engine.VULCAIN, Engine.RS68, Engine.S_IVB, Engine.SRB, Engine.P80, Engine.GEM60]
    _head_shapes = [HeadShape.CONE, HeadShape.ELLIPTICAL, HeadShape.SPHERE]

    _less_constrained = False

    def __init__(self):
        check_dependency()

        lc = self._less_constrained
        des_vars = [
            Integer(bounds=(1, 3)),  # Nr of stages

            Choice(options=list(range(6))),  # Engine type stage 1 [VULCAIN, RS68, S_IVB, SRB, P80, GEM60] [1]
            Integer(bounds=(1, 3)),  # Nr of engines stage 1
            Choice(options=list(range(6))),  # Engine type stage 2 [VULCAIN, RS68, S_IVB, SRB, P80, GEM60]
            Integer(bounds=(1, 3)),  # Nr of engines stage 2
            Choice(options=list(range(6))),  # Engine type stage 3 [VULCAIN, RS68, S_IVB, SRB, P80, GEM60]
            Integer(bounds=(1, 3)),  # Nr of engines stage 3

            Real(bounds=(0, 1) if lc else (0, 45)),  # Stage 1 length [m] [7]
            Real(bounds=(.1, .6) if lc else (0, 45)),  # Stage 2 length [m]
            Real(bounds=(.2, 1) if lc else (0, 45)),  # Stage 3 length [m]
            Real(bounds=(10, 11) if lc else (10, 20)),  # Overall length-to-diameter ratio [10]

            Choice(options=list(range(3))),  # Head shape [Cone, Elliptical, Semi-sphere] [11]
            Real(bounds=(28, 32) if lc else (15, 45)),  # Cone angle
            Real(bounds=(.15, .21) if lc else (.1, .25)),  # Ellipse length ratio
        ]

        super().__init__(des_vars, n_obj=2, n_ieq_constr=3 if lc else 2)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._correct_x_impute(x, is_active_out)

        lc = self._less_constrained
        rockets = self._get_rockets(x)
        for i, rocket in enumerate(rockets):
            perf = RocketEvaluator.evaluate(rocket)
            f_out[i, :] = (np.log10(perf.cost), -np.log10(max(1., perf.payload_mass)))
            if lc:
                g_out[i, :] = (perf.delta_structural, perf.delta_payload, perf.delta_delta_v)
            else:
                g_out[i, :] = (perf.delta_structural, perf.delta_payload)

    @classmethod
    def _get_rockets(cls, x: np.ndarray) -> List[Rocket]:
        lc = cls._less_constrained
        lc_stage1_bounds = [(15, 30), (15, 40), (15, 40)]

        rockets = []
        for i, xi in enumerate(x):
            n_stages = int(xi[0])
            stages = []
            prev_stage_length = 0
            for i_stage in range(n_stages):
                engine_type = cls._engines[int(xi[1+2*i_stage])]
                n_engines = int(xi[2+2*i_stage])

                stage_length = xi[7+i_stage]
                if lc:
                    if i_stage == 0:
                        stage1_bounds = lc_stage1_bounds[n_stages-1]
                        stage_length = stage_length*(stage1_bounds[1]-stage1_bounds[0]) + stage1_bounds[0]
                    elif i_stage > 0:
                        stage_length *= prev_stage_length

                stages.append(Stage(engines=[engine_type]*n_engines, length=stage_length))
                prev_stage_length = stage_length

            rockets.append(Rocket(
                stages=stages,
                head_shape=cls._head_shapes[int(xi[11])],
                cone_angle=xi[12],
                ellipse_l_ratio=xi[13],
                length_diameter_ratio=xi[10],
                max_q=200e3 if lc else 50e3,
                payload_density=2810,
                orbit_altitude=100e3 if lc else 400e3,
            ))

        return rockets

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        for n_stages in [1, 2, 3]:
            i_x_n_stages = x[:, 0] == n_stages

            is_active[i_x_n_stages, 1+2*n_stages:7] = False  # Engine selection inactiveness
            is_active[i_x_n_stages, 7+n_stages:10] = False  # Stage length inactiveness

        is_active[:, 12:14] = False
        is_active[x[:, 11] == 0, 12] = True  # Cone
        is_active[x[:, 11] == 1, 13] = True  # Ellipse

    def _get_n_valid_discrete(self) -> int:
        n_stages = np.ones((3,))
        for i in range(3):
            n_stages[i:] *= 6*3  # engine types * n_engines

        n_stages *= 3  # Head shapes

        return int(np.sum(n_stages))

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        # Correction is applied, so we calculate it from all_discrete_x
        return

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        n_dv = self.n_var
        x_stages = []
        for n_stages in range(3):
            opts_engines = (list(range(6)), list(range(1, 4)))
            x_engines = np.array(list(itertools.product(*(opts_engines*(n_stages+1)))), dtype=int)

            x_stage = np.zeros((x_engines.shape[0], n_dv))
            x_stage[:, 0] = n_stages+1
            x_stage[:, 1:1+x_engines.shape[1]] = x_engines
            x_stages.append(x_stage)

        x_stages = np.row_stack(x_stages)
        x_all = np.repeat(x_stages, 3, axis=0)
        x_all[:, [11]] = np.tile(np.array([np.arange(3)]).T, (x_stages.shape[0], 1))  # Head shape

        is_active_all = np.ones(x_all.shape, dtype=bool)
        self._correct_x(x_all, is_active_all)
        return x_all, is_active_all


class LCRocketArch(RocketArch):
    """
    Less constrained version of the rocket architecture problem.
    """
    _less_constrained = True


class RocketObj(enum.Enum):
    OBJ_COST = 1
    OBJ_PAYLOAD = 2
    OBJ_WEIGHTED = 3


class SOLCRocketArch(LCRocketArch):
    """
    Single-objective version of the less constrained rocket problem.
    """

    def __init__(self, obj: RocketObj = RocketObj.OBJ_COST):
        super().__init__()
        self.n_obj = 1
        self._obj = obj

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        f_mo = np.zeros((f_out.shape[0], 2))
        super()._arch_evaluate(x, is_active_out, f_mo, g_out, h_out, *args, **kwargs)

        if self._obj == RocketObj.OBJ_COST:
            f_out[:, 0] = f_mo[:, 0]
        elif self._obj == RocketObj.OBJ_PAYLOAD:
            f_out[:, 0] = f_mo[:, 1]
        elif self._obj == RocketObj.OBJ_WEIGHTED:
            f_out[:, 0] = f_mo[:, 0] + .5*f_mo[:, 1]
        else:
            raise RuntimeError(f'Unknown objective: {self._obj}')

    def __repr__(self):
        return f'{self.__class__.__name__}(obj={self._obj})'


if __name__ == '__main__':
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population
    from sb_arch_opt.sampling import HierarchicalSampling

    # problem = RocketArch()
    # x_pf = problem.pareto_set()
    # f_pf = problem.pareto_front()
    # problem = LCRocketArch()
    # problem = SOLCRocketArch(obj=RocketObj.OBJ_COST)
    problem = SOLCRocketArch(obj=RocketObj.OBJ_PAYLOAD)
    # problem = SOLCRocketArch(obj=RocketObj.OBJ_WEIGHTED)

    problem.plot_pf()
    # f_pf = problem.pareto_front()
    # f_so = f_pf[:, 0] + f_pf[:, 1]

    x_doe = HierarchicalSampling().do(problem, 200).get('X')
    pop = Evaluator().eval(problem, Population.new(X=x_doe))
    problem.get_population_statistics(pop, show=True)
