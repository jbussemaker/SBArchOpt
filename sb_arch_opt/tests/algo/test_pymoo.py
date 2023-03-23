import os
import pickle
import tempfile
import numpy as np
from typing import Optional
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.algo.pymoo_interface import *

from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.variable import Real, Integer
from pymoo.core.population import Population
from pymoo.problems.multi.zdt import ZDT1


def test_provision():
    ga = GA()
    provision_pymoo(ga)
    assert isinstance(ga.repair, ArchOptRepair)
    assert isinstance(ga.initialization.sampling, RepairedRandomSampling)


def test_nsga2(problem: ArchOptProblemBase):
    nsga2 = get_nsga2(pop_size=100)
    result = minimize(problem, nsga2, termination=('n_gen', 10), verbose=True, progress=True)
    pop = result.pop

    x_imp, _ = problem.correct_x(pop.get('X'))
    assert np.all(pop.get('X') == x_imp)


def test_termination(problem: ArchOptProblemBase):
    nsga2 = get_nsga2(pop_size=100)
    assert minimize(problem, nsga2, get_default_termination(problem, tol=1e-4), verbose=True, progress=True)


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
        self.n_stored_final = 0
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
        self._impute_x(x, is_active)

    def store_results(self, results_folder, final=False):
        if final:
            self.n_stored_final += 1
        else:
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
                assert len(nsga2.initialization.sampling) == 100

            minimize(problem, nsga2, termination=('n_gen', 3), copy_algorithm=False)
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_results.pkl'))
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_population.pkl'))
            assert os.path.exists(os.path.join(tmp_folder, 'pymoo_population.csv'))

            assert problem.n_eval == 3+2*i  # 3 for initial population, 2 for next because the first is a restart
            assert problem.n_stored == 6+5*i
            assert problem.n_stored_final == 2*(i+1)  # because pymoo calls result() twice in the end


def test_batch_storage_evaluator(problem: ArchOptProblemBase):
    with tempfile.TemporaryDirectory() as tmp_folder:
        pop = RepairedRandomSampling().do(problem, 110)
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
        assert np.all(pop_loaded.get('X') == pop.get('X'))
        assert np.all(pop_loaded.get('F') == pop.get('F'))
