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

This test suite contains the mixed-discrete, hierarchical, multi-objective turbofan architecture test problem
(subject to hidden constraints).
More information: https://github.com/jbussemaker/OpenTurbofanArchitecting
"""
import pickle
import numpy as np
from typing import *
import concurrent.futures
from pymoo.core.variable import Real, Integer, Choice
from sb_arch_opt.problems.hierarchical import HierarchyProblemBase

import os
os.environ['OPENMDAO_REQUIRE_MPI'] = 'false'  # Suppress OpenMDAO MPI import warnings
try:
    from open_turb_arch.architecting.architecting_problem import get_architecting_problem, load_pareto_front
    from open_turb_arch.architecting.problem import ArchitectingProblem
    from open_turb_arch.architecting.opt_defs import *
    from open_turb_arch.tests.examples.simple_problem import get_architecting_problem as get_simple_architecting_problem
    HAS_OPEN_TURB_ARCH = True
except ImportError:
    HAS_OPEN_TURB_ARCH = False

__all__ = ['HAS_OPEN_TURB_ARCH', 'SimpleTurbofanArch', 'RealisticTurbofanArch']


def check_dependency():
    if not HAS_OPEN_TURB_ARCH:
        raise RuntimeError('OpenTurbArch not installed: python setup.py install[ota]')


class OpenTurbArchProblemWrapper(HierarchyProblemBase):
    """
    Wrapper for an OpenTurbArch architecting problem: https://github.com/jbussemaker/OpenTurbofanArchitecting

    For more details see:
    [System Architecture Optimization: An Open Source Multidisciplinary Aircraft Jet Engine Architecting Problem](https://arc.aiaa.org/doi/10.2514/6.2021-3078)

    Available here:
    https://www.researchgate.net/publication/353530868_System_Architecture_Optimization_An_Open_Source_Multidisciplinary_Aircraft_Jet_Engine_Architecting_Problem
    """
    _force_get_discrete_rates = False
    default_enable_pf_calc = False
    _robust_correct_x = False  # Bug in the bleed offtake choice

    _data_folder = 'turbofan_data'

    def __init__(self, open_turb_arch_problem: 'ArchitectingProblem', n_parallel=None):
        check_dependency()
        self._problem = open_turb_arch_problem
        self.n_parallel = n_parallel
        self.verbose = False
        self.results_folder = None

        # open_turb_arch_problem.max_iter = 10  # Leads to a high failure rate: ~88% for the simple problem
        open_turb_arch_problem.max_iter = 30  # Used for the paper

        des_vars = []
        for dv in open_turb_arch_problem.free_opt_des_vars:
            if isinstance(dv, DiscreteDesignVariable):
                if dv.type == DiscreteDesignVariableType.INTEGER:
                    des_vars.append(Integer(bounds=(0, len(dv.values)-1)))

                elif dv.type == DiscreteDesignVariableType.CATEGORICAL:
                    des_vars.append(Choice(options=dv.values))
                else:
                    raise ValueError(f'Unknown discrete variable type: {dv.type}')

            elif isinstance(dv, ContinuousDesignVariable):
                des_vars.append(Real(bounds=dv.bounds))
            else:
                raise NotImplementedError

        n_obj = len(open_turb_arch_problem.opt_objectives)
        n_con = len(open_turb_arch_problem.opt_constraints)
        super().__init__(des_vars, n_obj=n_obj, n_ieq_constr=n_con)

        self._obj_factors = np.array([-1 if obj.dir == ObjectiveDirection.MAXIMIZE else 1
                                      for obj in open_turb_arch_problem.opt_objectives])
        self._con_factors = np.array([-1 if con.dir == ConstraintDirection.GREATER_EQUAL_THAN else 1
                                      for con in open_turb_arch_problem.opt_constraints])
        self._con_offsets = np.array([con.limit_value for con in open_turb_arch_problem.opt_constraints])

    def set_max_iter(self, max_iter: int):
        self._problem.max_iter = max_iter

    def might_have_hidden_constraints(self):
        return True

    def _get_n_valid_discrete(self) -> int:
        raise NotImplementedError

    def get_failure_rate(self) -> float:
        raise NotImplementedError

    def _is_conditionally_active(self) -> List[bool]:
        raise NotImplementedError

    def get_n_batch_evaluate(self) -> Optional[int]:
        return self.n_parallel

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

        if self._robust_correct_x:
            self._correct_x_impute(x, is_active_out)

        if self.n_parallel is None:
            results = [self._arch_evaluate_x(x[i, :]) for i in range(x.shape[0])]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = [executor.submit(self._arch_evaluate_x, x[i, :]) for i in range(x.shape[0])]
                concurrent.futures.wait(futures)
                results = [fut.result() for fut in futures]

        for i, (x_imputed, f, g, is_act) in enumerate(results):
            x[i, :] = x_imputed
            is_active_out[i, :] = is_act
            f_out[i, :] = np.array(f)*self._obj_factors
            g_out[i, :] = (np.array(g)-self._con_offsets)*self._con_factors

    def _arch_evaluate_x(self, x: np.ndarray):
        self._problem.verbose = self.verbose
        self._problem.save_results_folder = self.results_folder
        self._problem.save_results_combined = self.results_folder is not None

        x_imp, f, g, _ = self._problem.evaluate(self._convert_x(x))
        is_active = self._problem.get_last_is_active()
        return x_imp, f, g, is_active

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        n_correct = 3 if self._robust_correct_x else 1
        for i_corr in range(n_correct):
            if i_corr > 0:
                self.impute_x(x, is_active)

            for i in range(x.shape[0]):
                __, x[i, :] = self._problem.generate_architecture(self._convert_x(x[i, :]))
                is_active[i, :] = self._problem.get_last_is_active()

    def _convert_x(self, x) -> List[Union[float, int]]:
        mask = self.is_discrete_mask
        return [int(value) if mask[i] else float(value) for i, value in enumerate(x)]

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x_all_path = self._get_data_path(f'{self.__class__.__name__}_x_all.pkl')
        if os.path.exists(x_all_path):
            with open(x_all_path, 'rb') as fp:
                data = pickle.load(fp)

            if len(data) == 2:
                x_all, is_act_all = data
                if isinstance(x_all, np.ndarray) and isinstance(is_act_all, np.ndarray) and \
                        x_all.shape == is_act_all.shape and x_all.shape[0] == self._get_n_valid_discrete():
                    return x_all, is_act_all

        x_all, is_act_all = self.design_space.all_discrete_x_by_trial_and_imputation
        with open(x_all_path, 'wb') as fp:
            pickle.dump((x_all, is_act_all), fp)
        return x_all, is_act_all

    @classmethod
    def _get_data_path(cls, sub_path: str) -> str:
        path = os.path.join(os.path.dirname(__file__), cls._data_folder)
        os.makedirs(path, exist_ok=True)
        if sub_path is not None:
            path = os.path.join(path, sub_path)
        return path


class SimpleTurbofanArch(OpenTurbArchProblemWrapper):
    """
    Instantiation of the simple jet engine architecting problem:
    https://github.com/jbussemaker/OpenTurbofanArchitecting#simple-architecting-problem

    For more details see:
    [System Architecture Optimization: An Open Source Multidisciplinary Aircraft Jet Engine Architecting Problem](https://arc.aiaa.org/doi/10.2514/6.2021-3078)

    Available here:
    https://www.researchgate.net/publication/353530868_System_Architecture_Optimization_An_Open_Source_Multidisciplinary_Aircraft_Jet_Engine_Architecting_Problem
    """

    def __init__(self, n_parallel=None):
        check_dependency()
        super().__init__(get_simple_architecting_problem(), n_parallel=n_parallel)

    def _get_n_valid_discrete(self) -> int:
        n_valid_no_fan = 1
        n_valid_fan = 1

        # Gearbox, mixed nozzle choices
        n_valid_fan *= 2*2

        # Nr of shafts choice
        n_valid_no_fan = np.ones((3,))*n_valid_no_fan
        n_valid_fan = np.ones((3,))*n_valid_fan

        # Power and bleed offtakes
        n_valid_no_fan[1] *= 2*2
        n_valid_no_fan[2] *= 3*3
        n_valid_fan[1] *= 2*2
        n_valid_fan[2] *= 3*3

        return int(sum(n_valid_no_fan)+sum(n_valid_fan))

    def get_failure_rate(self) -> float:
        return .51  # Paper section IV.B

    def _is_conditionally_active(self) -> List[bool]:
        is_cond_active = [True]*self.n_var
        is_cond_active[0] = False  # Fan

        is_cond_active[3] = False  # Nr of shafts
        is_cond_active[4] = False  # Shaft 1 PR
        is_cond_active[7] = False  # Shaft 1 RPM

        return is_cond_active


class RealisticTurbofanArch(OpenTurbArchProblemWrapper):
    """
    Instantiation of the realistic jet engine architecting problem:
    https://github.com/jbussemaker/OpenTurbofanArchitecting#realistic-architecting-problem

    For more details see:
    [System Architecture Optimization: An Open Source Multidisciplinary Aircraft Jet Engine Architecting Problem](https://arc.aiaa.org/doi/10.2514/6.2021-3078)

    Available here:
    https://www.researchgate.net/publication/353530868_System_Architecture_Optimization_An_Open_Source_Multidisciplinary_Aircraft_Jet_Engine_Architecting_Problem
    """
    _robust_correct_x = True

    def __init__(self, n_parallel=None):
        check_dependency()
        super().__init__(get_architecting_problem(), n_parallel=n_parallel)

        x_pf, self.f_pf, _ = load_pareto_front()
        self.x_pf, _ = self.correct_x(x_pf)

    def _get_n_valid_discrete(self) -> int:
        n_valid_no_fan = 1
        n_valid_fan = 1

        # CRTF, gearbox, mixed nozzle choices
        n_valid_fan *= 2*2*2

        # Nr of shafts choice
        n_valid_no_fan = np.ones((3,))*n_valid_no_fan
        n_valid_fan = np.ones((3,))*n_valid_fan

        # ITB choice
        n_valid_no_fan[1:] *= 2
        n_valid_fan[1:] *= 2

        # Power and bleed offtakes
        n_valid_no_fan[1] *= 2*2
        n_valid_no_fan[2] *= 3*3
        n_valid_fan[1] *= 2*2
        n_valid_fan[2] *= 3*3

        # Intercooler choice
        n_valid_fan_include_ic = n_valid_fan.copy()
        n_valid_fan_include_ic *= 250
        n_valid_fan_include_ic[1] *= 2
        n_valid_fan_include_ic[2] *= 3
        n_valid_fan += n_valid_fan_include_ic

        return int(sum(n_valid_no_fan)+sum(n_valid_fan))

    def get_failure_rate(self) -> float:
        return .67  # Paper section IV.C

    def _is_conditionally_active(self) -> List[bool]:
        is_cond_active = [True]*self.n_var
        is_cond_active[0] = False  # Fan

        is_cond_active[4] = False  # Nr of shafts
        is_cond_active[5] = False  # Shaft 1 PR
        is_cond_active[6] = False  # Shaft 1 RPM

        return is_cond_active

    def _calc_pareto_front(self):
        return self.f_pf

    def _calc_pareto_set(self):
        return self.x_pf


if __name__ == '__main__':
    SimpleTurbofanArch().print_stats()
    # RealisticTurbofanArch().print_stats()

    # import pandas as pd
    # problem = RealisticTurbofanArch()
    # problem.print_stats()
    # x_all, is_act_all = problem.design_space.all_discrete_x_by_trial_and_imputation
    # dr = problem.get_discrete_rates(force=True)
    # with pd.ExcelWriter('real_turbofan.xlsx') as writer:
    #     pd.DataFrame(x_all, columns=[f'x{i}' for i in range(x_all.shape[1])]).to_excel(writer, sheet_name='x')
    #     pd.DataFrame(is_act_all, columns=[f'x{i}' for i in range(x_all.shape[1])]).to_excel(writer, sheet_name='is_act')
    #     dr.to_excel(writer, 'adr')

    # from pymoo.optimize import minimize
    # from sb_arch_opt.algo.pymoo_interface import get_nsga2
    # problem = SimpleTurbofanArch(n_parallel=4)
    # algo = get_nsga2(pop_size=2)
    # result = minimize(problem, algo, termination=('n_eval', 2))
    # print(result.pop.get('X'))
    # print(result.pop.get('F'))
    # print(result.pop.get('G'))
