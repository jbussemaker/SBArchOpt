import os
import copy
import pytest
import pickle
import tempfile
import numpy as np
import contextlib
from typing import Tuple
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from pymoo.core.variable import Real, Integer, Choice
from sb_arch_opt.problems.md_mo import *
from sb_arch_opt.problems.discrete import *
from sb_arch_opt.problems.continuous import *
from sb_arch_opt.problems.constrained import *
from sb_arch_opt.problems.hierarchical import *
from sb_arch_opt.problems.hidden_constraints import *
from sb_arch_opt.tests.algo.test_pymoo import CrashingProblem, CrashError
from sb_arch_opt.algo.pymoo_interface import load_from_previous_results
from pymoo.optimize import minimize
from pymoo.core.population import Population

from sb_arch_opt.algo.arch_sbo import *
from sb_arch_opt.algo.arch_sbo.algo import *
from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.arch_sbo.models import *
from sb_arch_opt.algo.arch_sbo.hc_strategy import *

if HAS_SMT:
    from smt.surrogate_models.rbf import RBF
    from smt.surrogate_models.krg import KRG
    from smt.surrogate_models.kpls import KPLS
    from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType

def check_dependency():
    if not HAS_SMT:
        return pytest.mark.skipif(not HAS_SMT, reason='SMT dependency not installed')
    else:
        return pytest.mark.skipif(not HAS_ARCH_SBO, reason='ArchSBO dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_ARCH_SBO


@check_dependency()
def test_arch_sbo_rbf(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    sbo = get_arch_sbo_rbf(init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12), verbose=True, progress=True)
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_rbf_seed(problem: ArchOptProblemBase):
    result = minimize(problem, get_arch_sbo_rbf(init_size=10), termination=('n_eval', 12), seed=42)
    x1 = result.pop.get('X')

    result = minimize(problem, get_arch_sbo_rbf(init_size=10), termination=('n_eval', 12), seed=42)
    x2 = result.pop.get('X')
    assert np.all(x1 == x2)


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
def test_arch_sbo_y(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    model = ModelFactory.get_kriging_model()
    infill = FunctionEstimateConstrainedInfill()
    sbo = get_sbo(model, infill, init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_ei(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    model = ModelFactory.get_kriging_model()
    infill = ExpectedImprovementInfill()
    sbo = get_sbo(model, infill, init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_ei_explicit():
    assert HAS_ARCH_SBO

    problem = Jenatton()
    assert problem.design_space.is_explicit()

    model = ModelFactory.get_kriging_model()
    infill = ExpectedImprovementInfill()
    sbo = get_sbo(model, infill, init_size=10)
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
def test_arch_sbo_hc_rf():
    assert HAS_ARCH_SBO

    hc_strategy = PredictionHCStrategy(RandomForestClassifier())

    problem = Mueller01()
    model = ModelFactory.get_kriging_model()
    infill = MinVariancePFInfill()
    sbo = get_sbo(model, infill, hc_strategy=hc_strategy, init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 11))
    assert len(result.pop) == 11


@check_dependency()
def test_arch_sbo_hc_md_gp():
    assert HAS_ARCH_SBO

    hc_strategy = PredictionHCStrategy(MDGPRegressor())

    problem = Mueller01()
    model = ModelFactory.get_kriging_model()
    infill = MinVariancePFInfill()
    sbo = get_sbo(model, infill, hc_strategy=hc_strategy, init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 11))
    assert len(result.pop) == 11


@check_dependency()
def test_arch_sbo_gp(problem: ArchOptProblemBase):
    assert HAS_ARCH_SBO

    _, n_batch = get_default_infill(problem)
    assert n_batch == 1

    sbo = get_arch_sbo_gp(problem, init_size=10)
    result = minimize(problem, sbo, termination=('n_eval', 11))
    assert len(result.pop) == 11


@check_dependency()
def test_arch_sbo_gp_seed(problem: ArchOptProblemBase):
    result = minimize(problem, get_arch_sbo_gp(problem, init_size=10), termination=('n_eval', 11), seed=42)
    x1 = result.pop.get('X')

    result = minimize(problem, get_arch_sbo_gp(problem, init_size=10), termination=('n_eval', 11), seed=42)
    x2 = result.pop.get('X')
    assert np.all(x1 == x2)


@check_dependency()
def test_arch_sbo_gp_pls():
    assert HAS_ARCH_SBO

    problem = HierarchicalGoldstein()
    sbo = get_arch_sbo_gp(problem, init_size=10, kpls_n_dim=3)
    result = minimize(problem, sbo, termination=('n_eval', 12))
    assert len(result.pop) == 12


@check_dependency()
def test_arch_sbo_gp_failing():
    assert HAS_ARCH_SBO

    problem = Mueller01()
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
def test_arch_sbo_gp_copy():
    assert HAS_ARCH_SBO

    problem = Branin()
    sbo = get_arch_sbo_gp(problem, init_size=10)

    problem, sbo = pickle.loads(pickle.dumps((problem, sbo)))
    sbo = copy.deepcopy(sbo)

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

            n_eval = 10+(i+1)
            result = minimize(problem, sbo, termination=('n_eval', n_eval))
            assert len(result.pop) == 10+(i+1)


@check_dependency()
def test_partial_restart():
    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(100):
            try:
                problem = CrashingProblem()
                pop = load_from_previous_results(problem, tmp_folder)
                n_evaluated = 0
                if i == 0:
                    assert pop is None
                else:
                    n_evaluated = 10*i

                    assert isinstance(pop, Population)
                    x = pop.get('X')
                    assert x.shape == (20*((i+1)//2), problem.n_var)

                    f = pop.get('F')
                    assert f.shape == (x.shape[0], problem.n_obj)
                    n_empty = np.sum(np.any(np.isnan(f), axis=1))
                    assert n_empty == x.shape[0]-n_evaluated

                sbo = get_arch_sbo_rbf(init_size=20, results_folder=tmp_folder)
                sbo.infill_size = 20

                sbo.initialize_from_previous_results(problem, result_folder=tmp_folder)
                assert sbo.evaluator.n_eval == n_evaluated
                result = minimize(problem, sbo, termination=('n_eval', 40))
                assert len(result.pop) == 40
                break

            except CrashError:
                pass


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@check_dependency()
def test_constraint_handling():
    problem = ArchCarside()
    assert problem.n_ieq_constr == 10

    for strategy, n_g_infill in [
        (MeanConstraintPrediction(), {problem.n_ieq_constr}),
        (MeanConstraintPrediction(aggregation=ConstraintAggregation.ELIMINATE), set(range(1, 5))),
        (MeanConstraintPrediction(aggregation=ConstraintAggregation.AGGREGATE), {1}),
        (ProbabilityOfFeasibility(), {problem.n_ieq_constr}),
        (ProbabilityOfFeasibility(min_pof=.25), {problem.n_ieq_constr}),
    ]:
        model = ModelFactory.get_kriging_model()
        infill = FunctionEstimateConstrainedInfill(constraint_strategy=strategy)
        sbo = get_sbo(model, infill, init_size=10)

        result = minimize(problem, sbo, termination=('n_eval', 12), copy_algorithm=False)
        assert infill.constraint_strategy.problem is problem
        assert infill.get_n_infill_constraints() in n_g_infill
        assert len(result.pop) == 12


class FailedXYRemovingSBO(SBOInfill):

    def _get_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_failed = np.any(~np.isfinite(y), axis=1)
        return x[~is_failed, :], y[~is_failed, :]


@check_dependency()
def test_invalid_training_set(problem: ArchOptProblemBase):
    assert HAS_SMT
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
    x = HierarchicalSampling().do(problem, 1000).get('X')
    assert np.all(np.round(np.min(x, axis=0)) == problem.xl)
    assert np.all(np.round(np.max(x, axis=0)) == problem.xu)

    md_norm = MixedDiscreteNormalization(problem.design_space)
    x_norm = md_norm.forward(x)
    assert np.all(np.round(np.min(x_norm, axis=0)) == [0, 0, 0, 0, 0])
    assert np.all(np.round(np.max(x_norm, axis=0)) == [1, 5, 3, 9, 2])

    x_abs = md_norm.backward(x_norm)
    assert np.all(x == x_abs)

@check_dependency()
def test_smt_krg_features():
    assert HAS_SMT
    n_pls = 2
    kwargs = dict(
        categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE,
        hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
    )
    pls_kwargs = dict(
        categorical_kernel=MixIntKernelType.GOWER,
        hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
    )

    def _try_model(problem: ArchOptProblemBase, pls: bool = False, cont_relax: bool = False, ignore_hierarchy=False,
                   throws_error=False, pls_cont=True):
        model_factory = ModelFactory(problem)
        normalization = model_factory.get_md_normalization()
        ds = model_factory.problem.design_space
        norm_ds_spec = model_factory.create_smt_design_space_spec(ds, md_normalize=True)

        if cont_relax and norm_ds_spec.is_mixed_discrete:
            cr_ds_spec = model_factory.create_smt_design_space_spec(ds, md_normalize=True, cont_relax=True)
            assert cr_ds_spec.design_space.is_all_cont
            model_ds = cr_ds_spec.design_space
        elif ignore_hierarchy:
            model_ds = model_factory.create_smt_design_space_spec(
                ds, md_normalize=True, ignore_hierarchy=True).design_space
        else:
            model_ds = norm_ds_spec.design_space

        if pls:
            pls_kwargs_ = pls_kwargs
            if not pls_cont:
                pls_kwargs_ = pls_kwargs.copy()
                pls_kwargs_.update(categorical_kernel=MixIntKernelType.EXP_HOMO_HSPHERE)
            model = KPLS(n_comp=n_pls, design_space=model_ds, **pls_kwargs_)
        else:
            model = KRG(design_space=model_ds, **kwargs)

        try:
            n_theta = ModelFactory.get_n_theta(problem, model)
            assert n_theta > 0

            model = MultiSurrogateModel(model)
            n_theta_multi = ModelFactory.get_n_theta(problem, model)
            assert n_theta_multi == n_theta*(problem.n_obj+problem.n_ieq_constr)

            x_train = HierarchicalSampling().do(problem, 50).get('X')
            y_train = problem.evaluate(x_train, return_as_dictionary=True)['F']
            x_test = HierarchicalSampling().do(problem, 200).get('X')

            model.set_training_values(normalization.forward(x_train), y_train)
            model.train()
            model.predict_values(normalization.forward(x_test))
            assert not throws_error
        except (TypeError, ValueError):
            assert throws_error
            return

        assert ModelFactory.get_n_theta(problem, model) == n_theta_multi
    # Continuous
    _try_model(Rosenbrock(), cont_relax=True)
    _try_model(Rosenbrock())
    _try_model(Rosenbrock(), pls=True, cont_relax=True)
    _try_model(Rosenbrock(), pls=True, ignore_hierarchy=True)
    _try_model(Rosenbrock(), pls=True)

    # Mixed-discrete (integer)
    _try_model(MDMORosenbrock(), cont_relax=True)
    _try_model(MDMORosenbrock())
    _try_model(MDMORosenbrock(), pls=True, cont_relax=True)
    _try_model(MDMORosenbrock(), pls=True, ignore_hierarchy=True)
    _try_model(MDMORosenbrock(), pls=True)

    # Mixed-discrete (categorical)
    _try_model(Halstrup04(), cont_relax=True)
    _try_model(Halstrup04())
    _try_model(Halstrup04(), pls=True, cont_relax=True)
    _try_model(Halstrup04(), pls=True, ignore_hierarchy=True)
    _try_model(Halstrup04(), pls=True)
    _try_model(Halstrup04(), pls=True, pls_cont=False)

    # Hierarchical (continuous conditional vars)
    _try_model(ZaeffererHierarchical(), cont_relax=True)
    _try_model(ZaeffererHierarchical())
    _try_model(ZaeffererHierarchical(), pls=True, cont_relax=True)
    _try_model(ZaeffererHierarchical(), pls=True, ignore_hierarchy=True)
    _try_model(ZaeffererHierarchical(), pls=True)

    # Hierarchical (integer conditional vars)
    _try_model(NeuralNetwork(), cont_relax=True)
    _try_model(NeuralNetwork())
    _try_model(NeuralNetwork(), pls=True, cont_relax=True)
    _try_model(NeuralNetwork(), pls=True, ignore_hierarchy=True)
    _try_model(NeuralNetwork(), pls=True)
    _try_model(NeuralNetwork(), pls=True, pls_cont=False)

    # Hierarchical (categorical conditional vars)
    _try_model(Jenatton(), cont_relax=True)
    _try_model(Jenatton() )
    _try_model(Jenatton(), pls=True, cont_relax=True)
    _try_model(Jenatton(), pls=True, ignore_hierarchy=True)
    _try_model(Jenatton(), pls=True)
    _try_model(Jenatton(), pls=True, ignore_hierarchy=True, pls_cont=False)
    _try_model(Jenatton(), pls=True, pls_cont=False)
