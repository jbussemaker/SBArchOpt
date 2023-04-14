import pytest
import tempfile
import numpy as np
from typing import Tuple
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from pymoo.core.variable import Real, Integer, Choice
from sb_arch_opt.algo.simple_sbo import *
from sb_arch_opt.algo.simple_sbo.algo import *
from sb_arch_opt.algo.simple_sbo.infill import *
from sb_arch_opt.algo.simple_sbo.models import *
from pymoo.optimize import minimize

check_dependency = lambda: pytest.mark.skipif(not HAS_SIMPLE_SBO, reason='Simple SBO dependencies not installed')


@check_dependency()
def test_simple_sbo_rbf(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_rbf(init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12), verbose=True, progress=True)
    assert len(result.pop) == 12


@check_dependency()
def test_simple_sbo_rbf_termination(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_rbf(init_size=10)
    termination = get_sbo_termination(n_max_infill=12, tol=1e-3)
    assert minimize(problem, sbo, termination=termination, verbose=True, progress=True)


@check_dependency()
def test_simple_sbo_rbf_failing(failing_problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_rbf(init_size=10)
    result = minimize(failing_problem, sbo, termination=('n_eval', 12), verbose=True, progress=True)
    assert len(result.pop) == 12


@check_dependency()
def test_simple_sbo_krg(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_krg(init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_simple_sbo_krg_y(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_krg(init_size=10, use_mvpf=False)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_simple_sbo_krg_ei(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    sbo = get_simple_sbo_krg(init_size=10, use_ei=True)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_store_results_restart(problem: ArchOptProblemBase):
    assert HAS_SIMPLE_SBO

    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(2):
            sbo = get_simple_sbo_rbf(init_size=10)
            sbo.store_intermediate_results(tmp_folder)
            sbo.initialize_from_previous_results(problem, tmp_folder)

            n_eval = 11 if i == 0 else 1
            result = minimize(problem, sbo, termination=('n_eval', n_eval))
            assert len(result.pop) == 10+(i+1)


class FailedXYRemovingSBO(SBOInfill):

    def _get_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_failed = np.any(~np.isfinite(y), axis=1)
        return x[~is_failed, :], y[~is_failed, :]


@check_dependency()
def test_invalid_training_set(problem: ArchOptProblemBase):
    from smt.surrogate_models.rbf import RBF
    sbo = FailedXYRemovingSBO(RBF(print_global=False), FunctionEstimateInfill(), pop_size=100, termination=100,
                              repair=ArchOptRepair()).algorithm(infill_size=1, init_size=10)
    sbo.setup(problem)

    for i in range(2):
        pop = sbo.ask()
        assert len(pop) == (10 if i == 0 else 1)
        sbo.evaluator.eval(problem, pop)
        pop.set('F', pop.get('F')*np.nan)
        sbo.tell(pop)

    sbo.ask()


class MDNormProblem(ArchOptProblemBase):

    def __init__(self):
        super().__init__(des_vars=[
            Real(bounds=(-5, 2)),
            Integer(bounds=[0, 5]),
            Integer(bounds=[-1, 2]),
            Integer(bounds=[1, 10]),
            Choice(options=[1, 2, 3]),
        ])

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        raise RuntimeError

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def test_md_normalization():
    problem = MDNormProblem()
    x = HierarchicalRandomSampling().do(problem, 1000).get('X')
    assert np.all(np.round(np.min(x, axis=0)) == problem.xl)
    assert np.all(np.round(np.max(x, axis=0)) == problem.xu)

    md_norm = MixedDiscreteNormalization(problem)
    x_norm = md_norm.forward(x)
    assert np.all(np.round(np.min(x_norm, axis=0)) == [0, 0, 0, 0, 0])
    assert np.all(np.round(np.max(x_norm, axis=0)) == [1, 5, 3, 9, 2])

    x_abs = md_norm.backward(x_norm)
    assert np.all(x == x_abs)
