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
import logging
import numpy as np
from typing import *
import concurrent.futures
from cached_property import cached_property
from pymoo.core.variable import Real, Integer, Choice
from sb_arch_opt.problems.hierarchical import HierarchyProblemBase
from sb_arch_opt.util import capture_log, get_cache_path

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

__all__ = ['HAS_OPEN_TURB_ARCH', 'SimpleTurbofanArch', 'RealisticTurbofanArch', 'SimpleTurbofanArchModel']

log = logging.getLogger('sb_arch_opt.turb')


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
    _sub_folder = None

    def __init__(self, open_turb_arch_problem: 'ArchitectingProblem', n_parallel=None):
        check_dependency()
        self._problem = open_turb_arch_problem
        self.n_parallel = n_parallel
        self.verbose = False
        self.results_folder = None
        self._x_pf = None
        self._f_pf = None
        self._g_pf = None
        self._models: Optional[dict] = None

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

        self.exclude_from_serialization = {'_models'}

    def set_max_iter(self, max_iter: int):
        self._problem.max_iter = max_iter

    def might_have_hidden_constraints(self):
        return True

    def _get_n_valid_discrete(self) -> int:
        raise NotImplementedError

    def _get_n_active_cont_mean(self) -> Optional[float]:
        return

    def _get_n_correct_discrete(self) -> Optional[int]:
        return

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        return

    def get_failure_rate(self) -> float:
        raise NotImplementedError

    def _is_conditionally_active(self) -> List[bool]:
        raise NotImplementedError

    def get_n_batch_evaluate(self) -> Optional[int]:
        return self.n_parallel

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):

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
        for i in range(x.shape[0]):
            __, x[i, :] = self._problem.generate_architecture(self._convert_x(x[i, :]))
            is_active[i, :] = self._problem.get_last_is_active()

    def _convert_x(self, x) -> List[Union[float, int]]:
        mask = self.is_discrete_mask
        return [int(value) if mask[i] else float(value) for i, value in enumerate(x)]

    def load_pareto_front(self):
        if self._x_pf is None:
            self._x_pf = np.load(self._get_data_path('eval_x_pf.npy'))
            self._f_pf = np.load(self._get_data_path('eval_f_pf.npy'))
            self._g_pf = np.load(self._get_data_path('eval_g_pf.npy'))
            assert self._x_pf.shape[1] == self.n_var
            assert self._f_pf.shape[1] == self.n_obj
            assert self._g_pf.shape[1] == self.n_ieq_constr
            assert self._x_pf.shape[0] == self._f_pf.shape[0]
            assert self._x_pf.shape[0] == self._g_pf.shape[0]

            # Sort by first objective dimension to ensure Pareto front and set points match
            # (because pymoo sorts the Pareto front but not the Pareto set)
            i_sorted = np.argsort(self._f_pf[:, 0])
            self._x_pf = self._x_pf[i_sorted, :]
            self._f_pf = self._f_pf[i_sorted, :]
            self._g_pf = self._g_pf[i_sorted, :]

        return self._x_pf, self._f_pf, self._g_pf

    def _calc_pareto_front(self):
        return self.load_pareto_front()[1]

    def _calc_pareto_set(self):
        return self.load_pareto_front()[0]

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x_all_path = self._get_data_path('x_all.pkl')
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

    def _load_evaluated(self):
        x = np.load(self._get_data_path('eval_x.npy'))
        f = np.load(self._get_data_path('eval_f.npy'))
        g = np.load(self._get_data_path('eval_g.npy'))
        assert x.shape[1] == self.n_var
        assert f.shape[1] == self.n_obj
        assert g.shape[1] == self.n_ieq_constr
        assert x.shape[0] == f.shape[0]
        assert x.shape[0] == g.shape[0]
        return x, f, g

    @cached_property
    def _g_from_x(self):
        # Get nr of constraints defined from metrics
        n_eval_g = len([con for metric in self._problem.constraints
                        for con in metric.get_opt_constraints(self._problem.choices)])

        # Determine which constraints are defined by choices
        g_is_from_x = np.zeros((self.n_ieq_constr,), dtype=bool)
        g_is_from_x[n_eval_g:] = True
        return g_is_from_x, n_eval_g

    @cached_property
    def _choice_g_props(self):
        return [(choice, choice.get_constraints() is not None, len(choice.get_design_variables()))
                for choice in self._problem.choices]

    def _ensure_models_trained(self):
        if self._models is None:
            self._train_models()

    def _train_models(self):
        import joblib
        from sklearn import ensemble
        model_data_dir = self._get_data_path('model_data')
        os.makedirs(model_data_dir, exist_ok=True)
        model_cache_dir = self._get_cache_path('model_data')
        os.makedirs(model_cache_dir, exist_ok=True)

        self._models = models = {}
        models_from_cache = set()

        def _get_model(name, x_, y_):
            def _inner():
                cache_path = f'{model_cache_dir}/{name}.pkl'
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as fp_:
                        model = pickle.load(fp_)
                    models_from_cache.add(name)
                    return model

                data_path = f'{model_data_dir}/{name}.pkl'
                if os.path.exists(data_path):
                    with open(data_path, 'rb') as fp_:
                        model = joblib.load(fp_)
                    models_from_cache.add(name)

                    with open(cache_path, 'wb') as fp_:
                        pickle.dump(model, fp_)
                    return model

                capture_log()
                model = ensemble.RandomForestRegressor(n_estimators=400)
                log.info(f'Training: {self.__class__.__name__}.{name} (xt = {x_.shape})')
                model.fit(x_, y_)
                return model

            mdl = _inner()

            # Fix old model version
            try:
                repr(mdl)
            except AttributeError:
                for est in mdl.estimators_:
                    est.monotonic_cst = None

            return mdl

        x, f, g = self._load_evaluated()
        is_failed = self.get_failed_points({'F': f, 'G': g})
        models['fails'] = _get_model('fails', x, is_failed.astype(float))

        x_ok = x[~is_failed, :]
        f = f[~is_failed, :]
        g = g[~is_failed, :]
        for i in range(f.shape[1]):
            models[f'f{i}'] = _get_model(f'f{i}', x_ok, f[:, i])

        g_from_x, _ = self._g_from_x
        for i in range(g.shape[1]):
            if not g_from_x[i]:
                models[f'g{i}'] = _get_model(f'g{i}', x_ok, g[:, i])

        for key, model_ in models.items():
            if key in models_from_cache:
                continue
            with open(f'{model_data_dir}/{key}.pkl', 'wb') as fp:
                joblib.dump(model_, fp, compress=9)

    def _check_pf_models(self):
        from sb_arch_opt.algo.pymoo_interface import get_nsga2
        from pymoo.optimize import minimize
        x_pf = self.pareto_set()
        f_pf = self.pareto_front()
        res = minimize(self, get_nsga2(pop_size=100), termination=('n_gen', 20))
        x_pf_surrogate = res.X
        f_pf_surrogate = res.F

        import matplotlib.pyplot as plt
        x, f, g = self._load_evaluated()
        n = x.shape[0]
        is_failed = self.get_failed_points({'F': f, 'G': g})
        is_failed_pred = self._models['fails'].predict(x) > .5
        false_pos = np.where(is_failed_pred & ~is_failed)[0]
        false_neg = np.where(~is_failed_pred & is_failed)[0]
        print(f'Failure prediction (n = {n}): false pos {len(false_pos)} ({100*len(false_pos)/n:.1f}%); '
              f'false neg {len(false_neg)} ({100*len(false_neg)/n:.1f}%)')

        for key, model in self._models.items():
            if key == 'fails':
                continue
            if key.startswith('f'):
                y = f[~is_failed, int(key[1:])]
            elif key.startswith('g'):
                y = g[~is_failed, int(key[1:])]
            else:
                continue
            y_pred = model.predict(x[~is_failed, :])
            plt.figure(), plt.title(f'Comparison: {key}')
            plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], '-r', linewidth=1)
            plt.scatter(y, y_pred, s=5, c='k')
            plt.xlabel('$y$'), plt.ylabel('$y_{predicted}$')

        if f_pf.shape[1] == 1:
            print(f'Best:\n{f_pf[0, 0]:.3g} @ {x_pf}')
            try:
                print(f'Surrogate model best:\n{f_pf_surrogate[0]:.3g} @ {x_pf_surrogate}')
                print(f'Diff:\n{f_pf_surrogate[0]-f_pf[0, 0]:.3g} @ {x_pf_surrogate-x_pf}')
            except TypeError:
                print(f'Surrogate model best (fail): {f_pf_surrogate[0]} @ {x_pf_surrogate}')
        else:
            plt.figure(), plt.title(f'Pareto Front comparison: {self.__class__.__name__}')
            plt.scatter(f_pf[:, 0], f_pf[:, 1], s=5, c='k', label='Original')
            plt.scatter(f_pf_surrogate[:, 0], f_pf_surrogate[:, 1], s=5, c='r', label='Predicted')
            plt.legend()
        plt.show()

    def _arch_evaluate_x_surrogate(self, x: np.ndarray):
        x_imp, is_active = self.correct_x(x)
        f = np.zeros((x.shape[0], self.n_obj))
        g = np.zeros((x.shape[0], self.n_ieq_constr))

        fails = self._models['fails'].predict(x_imp)
        is_failed: np.ndarray = fails > .5
        f[is_failed, :] = np.nan
        g[is_failed, :] = np.nan

        is_ok = ~is_failed
        x_ok = x_imp[is_ok, :]
        if x_ok.shape[0] > 0:
            for i in range(f.shape[1]):
                f[is_ok, i] = self._models[f'f{i}'].predict(x_ok)

        g_from_x, n_eval_g = self._g_from_x
        if np.any(g_from_x):
            for i in range(x.shape[0]):
                if is_ok[i]:
                    architecture, x_imp_ = self._problem.generate_architecture(self._convert_x(x_imp[i, :]))
                    _, x_full = self._problem.get_full_design_vector(x_imp_)

                    xii, i_g = 0, n_eval_g
                    for choice, has_g, n_dv in self._choice_g_props:
                        choice_dv = x_full[xii:xii+n_dv]

                        if has_g:
                            choice_g = choice.evaluate_constraints(
                                architecture, choice_dv, self._problem.analysis_problem, {})
                            g[i, i_g:i_g+len(choice_g)] = choice_g
                            i_g += len(choice_g)

                        xii += n_dv

            g[:, g_from_x] = (g[:, g_from_x]-self._con_offsets[g_from_x])*self._con_factors[g_from_x]

        if x_ok.shape[0] > 0:
            for i in range(g.shape[1]):
                if not g_from_x[i]:
                    g[is_ok, i] = self._models[f'g{i}'].predict(x_ok)

        return x_imp, f, g, is_active

    @classmethod
    def _get_data_path(cls, sub_path: str) -> str:
        path = os.path.join(os.path.dirname(__file__), cls._data_folder, cls._sub_folder)
        os.makedirs(path, exist_ok=True)
        if sub_path is not None:
            path = os.path.join(path, sub_path)
        return path

    @classmethod
    def _get_cache_path(cls, sub_path: str = None) -> str:
        path = get_cache_path(os.path.join('turbofan_cache', cls._sub_folder))
        os.makedirs(path, exist_ok=True)
        if sub_path is not None:
            path = os.path.join(path, sub_path)
        return path

    @classmethod
    def _get_models_cache_dir(cls):
        models_cache_dir = cls._get_data_path('model_data')
        os.makedirs(models_cache_dir, exist_ok=True)
        return models_cache_dir


class SimpleTurbofanArch(OpenTurbArchProblemWrapper):
    """
    Instantiation of the simple jet engine architecting problem:
    https://github.com/jbussemaker/OpenTurbofanArchitecting#simple-architecting-problem

    For more details see:
    [System Architecture Optimization: An Open Source Multidisciplinary Aircraft Jet Engine Architecting Problem](https://arc.aiaa.org/doi/10.2514/6.2021-3078)

    Available here:
    https://www.researchgate.net/publication/353530868_System_Architecture_Optimization_An_Open_Source_Multidisciplinary_Aircraft_Jet_Engine_Architecting_Problem
    """
    _sub_folder = 'simple'

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
    _sub_folder = 'realistic'

    def __init__(self, n_parallel=None, noise_obj=True):
        check_dependency()
        self.noise_obj = noise_obj

        arch_prob = get_architecting_problem()
        if not noise_obj:
            arch_prob._objectives = arch_prob._objectives[:2]
            arch_prob._opt_obj = None

        super().__init__(arch_prob, n_parallel=n_parallel)

        if not noise_obj:
            assert self.n_obj == 2
            assert len(self._obj_factors) == 2

    def get_original_pf(self):
        x_pf, f_pf, g_pf = load_pareto_front()
        x_pf, _ = self.correct_x(x_pf)
        x_pf, f_pf, g_pf = self._correct_pf(x_pf, f_pf, g_pf)
        return x_pf, f_pf, g_pf

    def _correct_pf(self, x_pf, f_pf, g_pf):
        if not self.noise_obj:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            f_pf = f_pf[:, :2]
            i_pf = NonDominatedSorting().do(f_pf, only_non_dominated_front=True)
            x_pf, f_pf, g_pf = x_pf[i_pf, :], f_pf[i_pf, :], g_pf[i_pf, :]

        return x_pf, f_pf, g_pf

    def load_pareto_front(self):
        if self._f_pf is None:
            n_obj = self.n_obj
            self.n_obj = 3

            self._x_pf, self._f_pf, self._g_pf = self._correct_pf(*super().load_pareto_front())

            self.n_obj = n_obj

        return self._x_pf, self._f_pf, self._g_pf

    def _load_evaluated(self):
        n_obj = self.n_obj
        self.n_obj = 3
        x, f, g = super()._load_evaluated()
        self.n_obj = n_obj
        return x, f, g

    def _arch_evaluate_x(self, x: np.ndarray):
        x_imp, f, g, is_active = super()._arch_evaluate_x(x)

        if not self.noise_obj:
            f = f[:2]

        return x_imp, f, g, is_active

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
        is_cond_active[8] = False  # Shaft 1 RPM

        return is_cond_active

    def __repr__(self):
        noise_obj = 'noise_obj=False' if not self.noise_obj else ''
        return f'{self.__class__.__name__}({noise_obj})'


class SimpleTurbofanArchModel(SimpleTurbofanArch):

    def __init__(self, train=True):
        super().__init__()
        if train:
            self._train_models()

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self._ensure_models_trained()
        x[:, :], f_out[:, :], g_out[:, :], is_active_out[:, :] = self._arch_evaluate_x_surrogate(x)

    def _arch_evaluate_x(self, x: np.ndarray):
        return self._arch_evaluate_x_surrogate(x)


# class RealisticTurbofanArchModel(RealisticTurbofanArch):
#
#     def __init__(self):
#         super().__init__()
#         self._train_models()
#
#     def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
#                        h_out: np.ndarray, *args, **kwargs):
#         x[:, :], f_out[:, :], g_out[:, :], is_active_out[:, :] = self._arch_evaluate_x_surrogate(x)
#
#     def _arch_evaluate_x(self, x: np.ndarray):
#         return self._arch_evaluate_x_surrogate(x)


if __name__ == '__main__':
    print(SimpleTurbofanArch().pareto_front())
    # SimpleTurbofanArch().print_stats()
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

    # SimpleTurbofanArchModel().print_stats()
    SimpleTurbofanArchModel()._check_pf_models()
