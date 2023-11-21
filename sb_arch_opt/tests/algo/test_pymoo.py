import os
import json
import pickle
import tempfile
import numpy as np
import pandas as pd
from typing import Optional
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.pymoo_interface.random_search import RandomSearchAlgorithm

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.variable import Real, Integer
from pymoo.core.population import Population
from pymoo.problems.multi.zdt import ZDT1


def test_provision():
    ga = GA()
    provision_pymoo(ga)
    assert isinstance(ga.repair, ArchOptRepair)
    assert isinstance(ga.initialization.sampling, HierarchicalSampling)


def test_nsga2(problem: ArchOptProblemBase):
    nsga2 = get_nsga2(pop_size=100)
    result = minimize(problem, nsga2, termination=('n_gen', 10), verbose=True, progress=True)
    pop = result.pop

    x_imp, _ = problem.correct_x(pop.get('X'))
    assert np.all(pop.get('X') == x_imp)


def test_termination(problem: ArchOptProblemBase):
    nsga2 = get_nsga2(pop_size=100)
    assert minimize(problem, nsga2, get_default_termination(problem, tol=1e-4), verbose=True, progress=True)


def test_seed(problem: ArchOptProblemBase):
    nsga2 = get_nsga2(pop_size=100)
    result1 = minimize(problem, nsga2, termination=('n_gen', 20), seed=42)

    nsga2 = get_nsga2(pop_size=100)
    result2 = minimize(problem, nsga2, termination=('n_gen', 20), seed=42)

    assert np.all(result1.pop.get('X') == result2.pop.get('X'))


def test_failing_evaluations(failing_problem: ArchOptProblemBase):
    nsga2 = get_nsga2(pop_size=100)
    assert minimize(failing_problem, nsga2, termination=('n_gen', 10), verbose=True, progress=True)


class DummyResultSavingProblem(ArchOptProblemBase):

    def __init__(self):
        self._problem = problem = ZDT1(n_var=5)
        var_types = [Real(bounds=(0, 1)) if i % 2 == 0 else Integer(bounds=(0, 9)) for i in range(problem.n_var)]
        super().__init__(var_types, n_obj=problem.n_obj)

        self.n_eval = 0
        self.n_stored = 0
        self.last_evaluated = None
        self.provide_previous_results = True

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        self.n_eval += 1
        self._correct_x(x, is_active_out)
        x_eval = x.copy()
        x_eval[:, self.is_discrete_mask] /= 9
        out = self._problem.evaluate(x_eval, return_as_dictionary=True)
        f_out[:, :] = out['F']
        self.last_evaluated = (x.copy(), is_active_out.copy(), f_out.copy())

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        is_active[:, -1] = x[:, 1] < 5
        self.impute_x(x, is_active)

    def _is_conditionally_active(self):
        is_cond_active = [False for _ in range(self.n_var)]
        is_cond_active[-1] = True
        return is_cond_active

    def store_results(self, results_folder):
        self.n_stored += 1

        assert self.last_evaluated is not None
        with open(os.path.join(results_folder, 'problem_last_pop.pkl'), 'wb') as fp:
            pickle.dump(self.last_evaluated, fp)

    def load_previous_results(self, results_folder) -> Optional[Population]:
        if not self.provide_previous_results:
            return
        path = os.path.join(results_folder, 'problem_last_pop.pkl')
        if not os.path.exists(path):
            return
        with open(path, 'rb') as fp:
            x, is_active, f = pickle.load(fp)
        return Population.new(X=x, F=f, is_active=is_active)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def test_store_results_restart():
    problem = DummyResultSavingProblem()

    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(5):
            nsga2 = get_nsga2(pop_size=100, results_folder=tmp_folder)
            assert isinstance(nsga2.evaluator, ArchOptEvaluator)
            nsga2.evaluator.n_batch = 500
            assert isinstance(nsga2.callback, ResultsStorageCallback)

            if i > 2:
                problem.provide_previous_results = False
            assert initialize_from_previous_results(nsga2, problem, tmp_folder) == (i > 0)
            if i > 0:
                assert isinstance(nsga2.initialization.sampling, Population)
                assert len(nsga2.initialization.sampling) == 100+2*100*i

            minimize(problem, nsga2, termination=('n_gen', 3), copy_algorithm=False, seed=42)
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_results.pkl'))
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_population.pkl'))
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_population.csv'))
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_population_cumulative.pkl'))
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_population_cumulative.csv'))

            assert problem.n_eval == 3+2*i  # 3 for initial population, 2 for next because the first is a restart
            assert problem.n_stored == 6+5*i

            n_cumulative = load_from_previous_results(problem, tmp_folder)
            assert len(n_cumulative) == 100+2*100*(i+1)


def test_batch_storage_evaluator(problem: ArchOptProblemBase):
    with tempfile.TemporaryDirectory() as tmp_folder:
        pop = HierarchicalSampling().do(problem, 110)
        assert pop.get('F').shape == (110, 0)

        evaluator = ArchOptEvaluator(results_folder=tmp_folder, n_batch=20)
        pop = evaluator.eval(problem, pop)
        assert len(pop) == 110
        assert pop.get('F').shape == (110, 2)

        pop_loaded = load_from_previous_results(problem, tmp_folder)
        assert np.all(pop_loaded.get('X') == pop.get('X'))
        assert np.all(pop_loaded.get('F') == pop.get('F'))


def test_doe_algo(problem: ArchOptProblemBase):
    with tempfile.TemporaryDirectory() as tmp_folder:
        doe_algo = get_doe_algo(doe_size=100)
        doe_algo.setup(problem)
        doe_algo.run()
        pop = doe_algo.pop
        assert len(pop) == 100
        assert not os.path.exists(os.path.join(tmp_folder, 'pymoo_population.csv'))

        doe_algo = get_doe_algo(doe_size=100, results_folder=tmp_folder)
        doe_algo.setup(problem)
        doe_algo.run()
        pop = doe_algo.pop
        assert len(pop) == 100
        assert os.path.exists(os.path.join(tmp_folder, 'pymoo_population.csv'))

        pop_loaded = load_from_previous_results(problem, tmp_folder)
        assert pop_loaded.get('X').shape == pop.get('X').shape
        assert pop_loaded.get('F').shape == pop.get('F').shape

        doe_algo2 = get_doe_algo(doe_size=100, results_folder=tmp_folder)
        initialize_from_previous_results(doe_algo2, problem, tmp_folder)
        assert doe_algo2.set_doe_size(problem, doe_size=200)
        doe_algo2.setup(problem)
        doe_algo2.run()
        assert doe_algo2.evaluator.n_eval == 200
        assert len(doe_algo2.pop) == 200

        doe_algo3 = get_doe_algo(doe_size=200, results_folder=tmp_folder)
        initialize_from_previous_results(doe_algo3, problem, tmp_folder)
        assert not doe_algo3.set_doe_size(problem, doe_size=150)
        doe_algo3.setup(problem)
        doe_algo3.run()
        assert doe_algo3.evaluator.n_eval == 200
        assert len(doe_algo3.pop) == 150


def test_doe_algo_seed(problem: ArchOptProblemBase):
    x_doe = []
    for seed in [42, None, 42]:
        doe_algo = get_doe_algo(doe_size=100)
        doe_algo.setup(problem, seed=seed)
        doe_algo.run()
        x_doe.append(doe_algo.pop.get('X'))

    assert np.any(x_doe[0] != x_doe[1])
    assert np.all(x_doe[0] == x_doe[2])


class CrashError(RuntimeError):
    pass


class CrashingProblem(DummyResultSavingProblem):

    def __init__(self, failed_evals=True):
        self.failed_evals = failed_evals
        super().__init__()
        self.i_eval = 0

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        super()._arch_evaluate(x, is_active_out, f_out, g_out, h_out, *args, **kwargs)

        if self.failed_evals:
            i_failed = np.arange(0, x.shape[0])[::2]
            f_out[i_failed, :] = np.nan
            g_out[i_failed, :] = np.nan

        self.i_eval += 1
        if self.i_eval > 1:
            raise CrashError

    def get_n_batch_evaluate(self) -> Optional[int]:
        return 10

    def store_results(self, results_folder):
        pass

    def load_previous_results(self, results_folder) -> Optional[Population]:
        pass


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

                nsga2 = get_nsga2(pop_size=20, results_folder=tmp_folder)
                initialize_from_previous_results(nsga2, problem, tmp_folder)
                assert nsga2.evaluator.n_eval == n_evaluated
                result = minimize(problem, nsga2, termination=('n_eval', 40))
                assert len(result.pop) == 40
                break

            except CrashError:
                pass


def test_df_storage(failing_problem: ArchOptProblemBase):
    problem = failing_problem

    doe_algo = get_doe_algo(doe_size=100)
    doe_algo.setup(problem)
    doe_algo.run()

    pop = doe_algo.pop
    assert np.any(np.isinf(pop.get('F')))  # Failed points
    assert not np.any(np.isnan(pop.get('F')))  # Unevaluated points

    df = ArchOptEvaluator.get_pop_as_df(pop)
    assert len(df) == len(pop)
    assert np.all(np.any(np.isinf(df), axis=1) == np.any(np.isinf(pop.get('F')), axis=1))

    pop2 = ArchOptEvaluator.get_pop_from_df(df)
    assert len(pop2) == len(pop)
    assert np.all(pop.get('X') == pop2.get('X'))
    assert np.all(pop.get('F') == pop2.get('F'))

    pop3 = ArchOptEvaluator.get_pop_from_df(pd.DataFrame.from_dict(json.loads(json.dumps(df.to_dict()))))
    assert len(pop3) == len(pop)
    assert np.all(pop.get('X') == pop3.get('X'))
    assert np.all(pop.get('F') == pop3.get('F'))

    with tempfile.TemporaryDirectory() as tmp_folder:
        problem = CrashingProblem()
        doe_algo = get_doe_algo(doe_size=30, results_folder=tmp_folder)
        doe_algo.setup(problem)
        try:
            doe_algo.run()
        except CrashError:
            pass

        pop = ArchOptEvaluator.load_pop(tmp_folder, cumulative=False)
        assert np.any(np.isinf(pop.get('F')))  # Failed points
        assert np.any(np.isnan(pop.get('F')))  # Unevaluated points

        pop2 = ArchOptEvaluator.get_pop_from_df(
            pd.DataFrame.from_dict(json.loads(json.dumps(ArchOptEvaluator.get_pop_as_df(pop).to_dict()))))
        assert len(pop2) == len(pop)
        assert np.all(pop.get('X') == pop2.get('X'))
        assert np.array_equal(pop.get('F'), pop2.get('F'), equal_nan=True)


def test_partial_doe_restart():
    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(100):
            try:
                problem = CrashingProblem()
                pop = load_from_previous_results(problem, tmp_folder)
                n_empty = 30
                if i == 0:
                    assert pop is None
                else:
                    assert isinstance(pop, Population)
                    x = pop.get('X')
                    assert np.all(np.isfinite(x))
                    assert x.shape == (30, problem.n_var)

                    f = pop.get('F')
                    assert f.shape == (30, problem.n_obj)
                    n_empty = np.sum(np.any(np.isnan(f), axis=1))
                    assert n_empty == 30-i*10

                doe_algo = get_doe_algo(doe_size=30, results_folder=tmp_folder)
                initialize_from_previous_results(doe_algo, problem, tmp_folder)
                assert doe_algo.evaluator.n_eval == 30-n_empty
                doe_algo.setup(problem)
                doe_algo.run()
                break

            except CrashError:
                pass

        assert doe_algo.evaluator.n_eval == 30

        pop = load_from_previous_results(problem, tmp_folder)
        assert len(pop) == 30
        n_empty = np.sum(np.any(np.isnan(pop.get('F')), axis=1))
        assert n_empty == 0


def test_partial_doe_restart_ask_tell():
    with tempfile.TemporaryDirectory() as tmp_folder:
        for i in range(100):
            try:
                problem = CrashingProblem()
                pop = load_from_previous_results(problem, tmp_folder)
                n_empty = 30
                if i == 0:
                    assert pop is None
                else:
                    assert isinstance(pop, Population)
                    x = pop.get('X')
                    assert np.all(np.isfinite(x))
                    assert x.shape == (30, problem.n_var)

                    f = pop.get('F')
                    assert f.shape == (30, problem.n_obj)
                    n_empty = np.sum(np.any(np.isnan(f), axis=1))
                    assert n_empty == 30-i*10

                doe_algo = get_doe_algo(doe_size=30, results_folder=tmp_folder)
                initialize_from_previous_results(doe_algo, problem, tmp_folder)
                assert doe_algo.evaluator.n_eval == 30-n_empty
                doe_algo.setup(problem)

                pop = doe_algo.ask()
                assert len(pop) == 30

                evaluator = doe_algo.evaluator
                assert isinstance(evaluator, ArchOptEvaluator)
                for _ in range(5):
                    pop_to_eval = evaluator.eval_pre(pop)

                for batch_pop in evaluator.iter_pop_batch(problem, pop_to_eval):
                    out = problem.evaluate(batch_pop.get('X'), return_as_dictionary=True)
                    evaluator.eval_apply_to_pop(batch_pop, out)
                    evaluator.eval_batch_post(problem, pop_to_eval, batch_pop)

                evaluator.eval_post(problem, pop)
                doe_algo.tell(pop)
                assert not doe_algo.has_next()

                break

            except CrashError:
                pass

        assert doe_algo.evaluator.n_eval == 30

        pop = load_from_previous_results(problem, tmp_folder)
        assert len(pop) == 30
        n_empty = np.sum(np.any(np.isnan(pop.get('F')), axis=1))
        assert n_empty == 0


def test_random_search(problem: ArchOptProblemBase):
    rs = RandomSearchAlgorithm(n_init=10)
    result = minimize(problem, rs, termination=('n_eval', 100))
    assert len(result.pop) == 100
