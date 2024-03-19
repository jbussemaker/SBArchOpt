import os
import pytest
import tempfile
import numpy as np
from sb_arch_opt.sampling import *
from sb_arch_opt.problems.turbofan_arch import *
from sb_arch_opt.algo.pymoo_interface import get_nsga2
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.initialization import Initialization

def check_dependency():
    return pytest.mark.skipif(not HAS_OPEN_TURB_ARCH, reason='Turbofan arch dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_OPEN_TURB_ARCH


@check_dependency()
def test_simple_problem():
    problem = SimpleTurbofanArch()
    problem.print_stats()

    assert len(HierarchicalExhaustiveSampling(n_cont=1).do(problem, 0)) == problem._get_n_valid_discrete()

    problem.get_discrete_rates(force=True, show=True)

    x_all, is_act_all = problem.design_space.all_discrete_x_by_trial_and_imputation
    assert np.all(problem.is_conditionally_active == np.any(~is_act_all, axis=0))
    x_all_corr, is_act_all_corr = problem.correct_x(x_all)
    assert np.all(x_all_corr == x_all)
    assert np.all(is_act_all_corr == is_act_all)

    x_all, is_act_all = problem.all_discrete_x
    assert is_act_all is not None
    x_all_corr, is_act_all_corr = problem.correct_x(x_all)
    assert np.all(x_all_corr == x_all)
    assert np.all(is_act_all_corr == is_act_all)

    x = HierarchicalSampling().do(problem, 1000).get('X')
    x_corr, is_act = problem.correct_x(x)
    assert np.all(x_corr == x)
    x_corr2, is_act2 = problem.correct_x(x_corr)
    assert np.all(x_corr2 == x_corr)
    assert np.all(is_act2 == is_act)

    x_pf = problem.pareto_set()
    assert len(x_pf) == 1
    f_pf = problem.pareto_front()
    assert len(f_pf) == 1

    assert problem._load_evaluated()

    f_eval = problem.evaluate(x_pf[[0], :], return_as_dictionary=True)['F']
    assert np.all(np.isfinite(f_eval))
    assert np.all(np.abs(f_eval[0, :] - f_pf[0, :]) < 1e-3)


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@check_dependency()
def test_simple_problem_eval():
    with tempfile.TemporaryDirectory() as tmp_folder:
        problem = SimpleTurbofanArch(n_parallel=2)
        algo = get_nsga2(pop_size=2, results_folder=tmp_folder)

        algo.initialization = Initialization(Population.new(X=np.array([
            [1, 12.34088028, 1.293692084, 2, 49.95762965, 0.333333333, 0.333333333, 14857.31833, 16919.26153, 8325.806323, 0, 3, 0, 0, 0],
            [0, 7.25, 1.45, 1, 23.04752703, 0.176817974, 0, 14412.5685, 14436.53056, 10500, 0, 3, 0, 1, 0],
        ])))

        result = minimize(problem, algo, termination=('n_eval', 2))
        f, g = result.pop.get('F'), result.pop.get('G')
        assert np.all(np.abs(f[0, :]-np.array([7.038242017])) < 1e-2)
        assert np.all(np.abs(g[0, :]-np.array([-0.219864891, -0.566666667, -11.61994603, -11.61994603, -11.61994603])) < 1e-2)
        assert np.isinf(f[1, 0])
        assert np.all(np.isinf(g[1, :]))


@check_dependency()
def test_simple_problem_model():
    problem = SimpleTurbofanArchModel()
    problem.print_stats()

    assert len(HierarchicalExhaustiveSampling(n_cont=1).do(problem, 0)) == problem._get_n_valid_discrete()

    problem.get_discrete_rates(force=True, show=True)

    x_all, is_act_all = problem.all_discrete_x
    assert is_act_all is not None
    out = problem.evaluate(x_all, return_as_dictionary=True)
    is_failed = np.where(problem.get_failed_points(out))[0]
    assert len(is_failed) == 67

    x_model_best = np.array([[1.00000000e+00, 1.24341485e+01, 1.31267687e+00, 2.00000000e+00,
                              5.71184472e+01, 5.00000000e-01, 5.00000000e-01, 8.39151709e+03,
                              1.05000000e+04, 7.85928873e+03, 1.00000000e+00, 3.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    out = problem.evaluate(x_model_best, return_as_dictionary=True)
    assert out['F'][0, 0] == pytest.approx(6.79, abs=1e-2)

    x = HierarchicalSampling().do(problem, 1000).get('X')
    problem.evaluate(x, return_as_dictionary=True)


@check_dependency()
def test_realistic_problem():
    problem = RealisticTurbofanArch()
    problem.print_stats()

    assert problem._get_n_valid_discrete() == 142243
    problem.get_discrete_rates(show=True)  # Takes several minutes
    x_all, is_act_all = problem.all_discrete_x
    assert x_all.shape[0] == problem.get_n_valid_discrete()

    i_random = np.random.choice(x_all.shape[0], 1000, replace=False)
    x_all_corr, is_act_all_corr = problem.correct_x(x_all[i_random])
    assert np.all(x_all_corr == x_all[i_random])
    assert np.all(is_act_all_corr == is_act_all[i_random])

    x = HierarchicalSampling().do(problem, 100).get('X')
    x_corr, is_act = problem.correct_x(x)
    assert np.all(x_corr == x)
    x_corr2, is_act2 = problem.correct_x(x_corr)
    assert np.all(x_corr2 == x_corr)
    assert np.all(is_act2 == is_act)

    f_pf = problem.pareto_front()
    x_pf = problem.pareto_set()
    x_pf_corr, is_act_pf = problem.correct_x(x_pf)
    assert np.all(x_pf_corr == x_pf)
    assert not np.all(is_act_pf)
    assert problem._load_evaluated()

    f_eval = problem.evaluate(x_pf[[0], :], return_as_dictionary=True)['F']
    assert np.all(np.isfinite(f_eval))
    assert np.all(np.abs(f_eval[0, :] - f_pf[0, :]) < 1e-3)


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@check_dependency()
def test_realistic_problem_2obj():
    problem = RealisticTurbofanArch(noise_obj=False)
    assert problem.n_obj == 2
    problem.print_stats()

    assert problem._get_n_valid_discrete() == 142243
    x_all, is_act_all = problem.all_discrete_x
    assert x_all.shape[0] == problem.get_n_valid_discrete()

    f_pf = problem.pareto_front()
    assert f_pf.shape[1] == 2
    x_pf = problem.pareto_set()
    assert f_pf.shape[0] == x_pf.shape[0]
    x_pf_corr, is_act_pf = problem.correct_x(x_pf)
    assert np.all(x_pf_corr == x_pf)
    assert not np.all(is_act_pf)
    assert problem._load_evaluated()

    f_eval = problem.evaluate(x_pf[[0], :], return_as_dictionary=True)['F']
    assert np.all(np.isfinite(f_eval))
    assert np.all(np.abs(f_eval[0, :] - f_pf[0, :]) < 1e-3)
