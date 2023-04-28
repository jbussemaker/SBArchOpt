import pytest
import tempfile
import numpy as np
from typing import Tuple
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from pymoo.core.variable import Real, Integer, Choice
from sb_arch_opt.algo.arch_sbo import *
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.problems.constrained import *
from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.arch_sbo.models import *
from pymoo.optimize import minimize

check_dependency = lambda: pytest.mark.skipif(not HAS_ARCH_SBO, reason='ArchSBO dependencies not installed')


@check_dependency()
def test_arch_sbo_rbf(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    sbo = get_arch_sbo_rbf(init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12), verbose=True, progress=True)
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_rbf_termination(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    sbo = get_arch_sbo_rbf(init_size=10)
    termination = get_sbo_termination(n_max_infill=12, tol=1e-3)
    assert minimize(problem, sbo, termination=termination, verbose=True, progress=True)


@check_dependency()
def test_arch_sbo_rbf_failing(failing_problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    sbo = get_arch_sbo_rbf(init_size=10)
    result = minimize(failing_problem, sbo, termination=('n_eval', 12), verbose=True, progress=True)
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_krg(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    sbo = get_arch_sbo_krg(init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_krg_y(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    sbo = get_arch_sbo_krg(init_size=10, use_mvpf=False)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_krg_ei(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    sbo = get_arch_sbo_krg(init_size=10, use_ei=True)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_mvpf():
    model = ModelFactory.get_kriging_model()
    infill = MinVariancePFInfill()
    sbo = get_sbo(model, infill, init_size=10)
    result = minimize(MOHimmelblau(), sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_gp(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    _, n_batch = get_default_infill(problem)
    assert n_batch == 1

    sbo = get_arch_sbo_gp(problem, init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_gp_batch(problem: ArchOptProblemBase):
    _, n_batch = get_default_infill(problem, n_parallel=5)
    assert n_batch == 5

    sbo = get_arch_sbo_gp(problem, init_size=10, n_parallel=5)
    result = minimize(problem, sbo, termination=('n_gen', 2))
    assert len(result.pop) == 15


@check_dependency()
def test_arch_sbo_gp_high_dim():
    assert HAS_ARCH_SBO

    problem = MOZDT1()
    sbo = get_arch_sbo_gp(problem, init_size=10, kpls_n_dim=5)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_store_results_restart(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(2):
            sbo = get_arch_sbo_rbf(init_size=10)
            sbo.store_intermediate_results(tmp_folder)
            sbo.initialize_from_previous_results(problem, tmp_folder)

            n_eval = 11 if i == 0 else 1
            result = minimize(problem, sbo, termination=('n_eval', n_eval))
            assert len(result.pop) == 10+(i+1)


@check_dependency()
def test_constraint_handling():
    problem = ArchCantileveredBeam()
    assert problem.n_ieq_constr > 0

    for strategy in [MeanConstraintPrediction(), ProbabilityOfFeasibility(), ProbabilityOfFeasibility(min_pof=.25)]:
        model = ModelFactory.get_kriging_model()
        infill = FunctionEstimateConstrainedInfill(constraint_strategy=strategy)
        sbo = get_sbo(model, infill, init_size=10)

        result = minimize(problem, sbo, termination=('n_eval', 12), copy_algorithm=False)
        assert infill.constraint_strategy.problem is problem
        assert infill.get_n_infill_constraints() == problem.n_ieq_constr
        assert len(result.pop) == 12


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

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def test_md_normalization():
    problem = MDNormProblem()
    x = HierarchicalRandomSampling().do(problem, 1000).get('X')
    assert np.all(np.round(np.min(x, axis=0)) == problem.xl)
    assert np.all(np.round(np.max(x, axis=0)) == problem.xu)

    md_norm = MixedDiscreteNormalization(problem.design_space)
    x_norm = md_norm.forward(x)
    assert np.all(np.round(np.min(x_norm, axis=0)) == [0, 0, 0, 0, 0])
    assert np.all(np.round(np.max(x_norm, axis=0)) == [1, 5, 3, 9, 2])

    x_abs = md_norm.backward(x_norm)
    assert np.all(x == x_abs)
