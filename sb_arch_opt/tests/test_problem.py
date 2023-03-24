import os
import pytest
import itertools
import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from sb_arch_opt.problems.problems_base import *
from pymoo.core.variable import Real, Integer, Binary, Choice


def test_init_no_vars():
    problem = ArchOptProblemBase([], n_obj=2, n_ieq_constr=2)
    assert problem.n_var == 0
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 2

    assert problem.get_n_declared_discrete() == 1
    assert np.isnan(problem.get_imputation_ratio())
    problem.print_stats()


def test_init_vars():
    problem = ArchOptProblemBase([
        Real(bounds=(1, 5)),
        Integer(bounds=(1, 4)),
        Binary(),
        Choice(options=['A', 'B', 'C']),
    ])
    assert problem.n_var == 4
    assert problem.n_obj == 1

    assert np.all(problem.xl == [1, 1, 0, 0])
    assert np.all(problem.xu == [5, 4, 1, 2])
    assert np.all(problem.is_cat_mask == [False, False, False, True])
    assert np.all(problem.is_int_mask == [False, True, True, False])
    assert np.all(problem.is_discrete_mask == [False, True, True, True])
    assert np.all(problem.is_cont_mask == [True, False, False, False])

    assert problem.get_n_declared_discrete() == 4*2*3


def test_rounding():
    problem = ArchOptProblemBase([Integer(bounds=(0, 5)), Integer(bounds=(-1, 1)), Integer(bounds=(2, 4))])
    assert np.all(problem.is_discrete_mask)
    x = np.array(list(itertools.product(np.linspace(0, 5, 20), np.linspace(-1, 1, 20), np.linspace(2, 4, 20))))
    problem._correct_x_discrete(x)

    x1, x1_counts = np.unique(x[:, 0], return_counts=True)
    assert np.all(x1 == [0, 1, 2, 3, 4, 5])
    x1_counts = x1_counts/np.sum(x1_counts)
    assert np.all(np.abs(x1_counts - np.mean(x1_counts)) <= .05)

    x2, x2_counts = np.unique(x[:, 1], return_counts=True)
    assert np.all(x2 == [-1, 0, 1])
    x2_counts = x2_counts/np.sum(x2_counts)
    assert np.all(np.abs(x2_counts - np.mean(x2_counts)) <= .05)

    x3, x3_counts = np.unique(x[:, 2], return_counts=True)
    assert np.all(x3 == [2, 3, 4])
    x3_counts = x3_counts/np.sum(x3_counts)
    assert np.all(np.abs(x3_counts - np.mean(x3_counts)) <= .05)


def test_correct_x(problem: ArchOptProblemBase):
    assert problem.n_var == 5
    assert np.all(problem.is_discrete_mask == [False, True, False, True, False])
    assert np.all(problem.is_int_mask == [False, False, False, True, False])
    assert np.all(problem.is_cat_mask == [False, True, False, False, False])
    assert np.all(problem.is_cont_mask == [True, False, True, False, True])

    x, is_active = problem.correct_x(np.array([
        [0, 0.1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 6.9, 0, 0, 1],
    ]))
    assert np.all(x == [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 7, 0, 0, .5],
    ])
    assert np.all(is_active == [
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, False],
    ])


def test_repair(problem: ArchOptProblemBase):
    assert isinstance(problem.get_repair(), ArchOptRepair)

    for as_pop in [False, True]:
        repair = ArchOptRepair()
        x = np.array([
            [0, 0.1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 6.9, 0, 0, 1],
        ])
        if as_pop:
            x = repair.do(problem, Population.new(X=x)).get('X')
        else:
            x = repair.do(problem, x)
        assert np.all(x == [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 7, 0, 0, .5],
        ])
        assert np.all(repair.latest_is_active == [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, False],
        ])


def test_imputation_ratio(problem: ArchOptProblemBase, discrete_problem: ArchOptProblemBase):
    assert problem.get_n_declared_discrete() == 10*10
    assert problem.get_n_valid_discrete() == 10 * 10
    assert problem.get_imputation_ratio() == 1
    problem.print_stats()

    x_discrete, is_active_discrete = problem.all_discrete_x
    assert x_discrete.shape[0] == problem.get_n_valid_discrete()
    assert is_active_discrete.shape[0] == problem.get_n_valid_discrete()
    assert np.all(~LargeDuplicateElimination.eliminate(x_discrete))

    assert discrete_problem.get_n_declared_discrete() == 10*10
    assert discrete_problem.get_n_valid_discrete() == 10 * 5 + 5
    assert discrete_problem.get_imputation_ratio() == 1/.55
    discrete_problem.print_stats()

    x_discrete, is_active_discrete = discrete_problem.all_discrete_x
    assert x_discrete.shape[0] == discrete_problem.get_n_valid_discrete()
    assert is_active_discrete.shape[0] == discrete_problem.get_n_valid_discrete()
    assert np.all(~LargeDuplicateElimination.eliminate(x_discrete))


def test_evaluate(problem: ArchOptProblemBase):
    assert problem.n_var == 5

    assert problem.is_cat_mask[1]
    x = np.zeros((4, 2))
    x[:, 1] = np.array([0, 1, 2, 3])
    cat_values = problem.get_categorical_values(x, 1)
    assert np.all(cat_values == ['9', '8', '7', '6'])
    assert np.all((cat_values == '9') == (x[:, 1] == 0))

    for use_evaluator in [False, True]:
        x = np.array([
            [0, 0.1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 6.9, 0, 0, 1],
        ])
        if use_evaluator:
            pop = Evaluator().eval(problem, Population.new(X=x))
            x_out, is_active, f = pop.get('X'), pop.get('is_active'), pop.get('F')
        else:
            out = problem.evaluate(x, return_as_dictionary=True)
            x_out, is_active, f = out['X'], out['is_active'], out['F']
        assert np.all(x_out == [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 7, 0, 0, .5],
        ])
        assert np.all(is_active == [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, False],
        ])
        assert np.all(f == [
            [0, 1],
            [0, 3.25],
            [0, 3.875],
        ])


def test_large_duplicate_elimination():
    x = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, .1],
        [0, 2, 0],
        [0, 2, 0],
        [1, 0, 0],
        [1, 0, 0],
    ])
    pop = LargeDuplicateElimination().do(Population.new(X=x))
    assert len(pop) == 5

    pop = LargeDuplicateElimination().do(Population.new(X=x[:2, :]), Population.new(X=x[2:, :]), to_itself=False)
    assert len(pop) == 1

    pop = LargeDuplicateElimination().do(Population.new(X=x[:5, :]), Population.new(X=x[5:, :]))
    assert len(pop) == 3

    n, m = 4, 7
    x = np.array(list(itertools.product(*[list(range(n)) for _ in range(m)])))
    x = np.repeat(x, 2, axis=0)
    x = x[np.random.permutation(np.arange(x.shape[0])), :]
    assert x.shape == (2*n**m, m)
    pop = LargeDuplicateElimination().do(Population.new(X=x))
    assert len(pop) == n**m


def test_repaired_exhaustive_sampling(problem: ArchOptProblemBase):
    for has_cheap in [True, False]:
        problem.set_provide_all_x(has_cheap)

        sampling_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(problem)
        assert len(sampling_values) == 5
        assert np.prod([len(values) for values in sampling_values]) == 12500
        assert HierarchicalExhaustiveSampling.get_n_sample_exhaustive(problem) == 12500
        assert HierarchicalExhaustiveSampling.has_cheap_all_x_discrete(problem) == has_cheap

        sampling = HierarchicalExhaustiveSampling()
        assert isinstance(sampling._repair, ArchOptRepair)
        x = sampling.do(problem, 100).get('X')
        assert x.shape == (7500, 5)
        assert np.unique(x, axis=0).shape[0] == 7500
        problem.evaluate(x)

        x_imp, _ = problem.correct_x(x)
        assert np.all(x_imp == x)


class HierarchicalDummyProblem(ArchOptProblemBase):

    def __init__(self, n=4):
        self._n = n
        des_vars = []
        for _ in range(n):
            des_vars.append(Binary())
        for _ in range(n):
            des_vars.append(Real(bounds=(0, 1)))
        super().__init__(des_vars)

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        n = self._n
        for i, xi in enumerate(x):
            x_sel = xi[:n]
            is_one = np.where(x_sel == 1)[0]
            if len(is_one) > 0:
                j = is_one[0]
                is_active[i, j+1:n] = False
                is_active[i, n+j:] = False

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def test_repaired_exhaustive_sampling_hierarchical():
    problem = HierarchicalDummyProblem()
    x = HierarchicalExhaustiveSampling(n_cont=2).do(problem, 0).get('X')
    assert x.shape == (31, 8)  # 2**(n+1)-1

    x_imp, _ = problem.correct_x(x)
    assert np.all(x_imp == x)


def test_repaired_exhaustive_sampling_hierarchical_large():
    n = 12
    problem = HierarchicalDummyProblem(n=n)
    x = HierarchicalExhaustiveSampling(n_cont=2).do(problem, 0).get('X')
    assert x.shape == (2**(n+1)-1, n*2)

    x_imp, _ = problem.correct_x(x)
    assert np.all(x_imp == x)


def test_repaired_lhs_sampling(problem: ArchOptProblemBase):

    sampling = HierarchicalLatinHypercubeSampling()
    assert isinstance(sampling._repair, ArchOptRepair)
    x = sampling.do(problem, 1000).get('X')
    assert x.shape == (1000, 5)
    assert np.unique(x, axis=0).shape[0] == 1000
    problem.evaluate(x)

    x_imp, _ = problem.correct_x(x)
    assert np.all(x_imp == x)

    init = get_init_sampler()
    x = init.do(problem, 1000).get('X')
    assert x.shape == (1000, 5)


def test_sobol_sampling():
    for _ in range(100):
        i = HierarchicalRandomSampling._sobol_choice(5, 10, replace=True)
        assert len(i) == 5
        assert 0 <= np.min(i) <= np.max(i) < 10
        assert len(np.unique(i)) <= 10

        i = HierarchicalRandomSampling._sobol_choice(10, 5, replace=True)
        assert len(i) == 10
        assert 0 <= np.min(i) <= np.max(i) < 5
        assert len(np.unique(i)) <= 5

        i = HierarchicalRandomSampling._sobol_choice(5, 10, replace=False)
        assert len(i) == 5
        assert 0 <= np.min(i) <= np.max(i) < 10
        assert len(np.unique(i)) == len(i)

    with pytest.raises(ValueError):
        HierarchicalRandomSampling._sobol_choice(10, 5, replace=False)


def test_repaired_random_sampling(problem: ArchOptProblemBase):

    sampling_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(problem)
    assert len(sampling_values) == 5
    assert np.prod([len(values) for values in sampling_values]) == 12500

    for sobol in [False, True]:
        sampling = HierarchicalRandomSampling(sobol=sobol)
        assert isinstance(sampling._repair, ArchOptRepair)
        x = sampling.do(problem, 1000).get('X')
        assert x.shape == (1000, 5)
        assert np.unique(x, axis=0).shape[0] == 1000
        problem.evaluate(x)

        x_imp, _ = problem.correct_x(x)
        assert np.all(x_imp == x)


def test_repaired_random_sampling_non_exhaustive(problem: ArchOptProblemBase):

    sampling_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(problem)
    assert len(sampling_values) == 5
    assert np.prod([len(values) for values in sampling_values]) == 12500

    limit = HierarchicalRandomSampling._n_comb_gen_all_max
    HierarchicalRandomSampling._n_comb_gen_all_max = 10

    for sobol in [False, True]:
        sampling = HierarchicalRandomSampling(sobol=sobol)
        assert isinstance(sampling._repair, ArchOptRepair)
        x = sampling.do(problem, 1000).get('X')
        assert x.shape == (1000, 5)
        assert np.unique(x, axis=0).shape[0] == 1000
        problem.evaluate(x)

        x_imp, _ = problem.correct_x(x)
        assert np.all(x_imp == x)

    HierarchicalRandomSampling._n_comb_gen_all_max = limit


def test_cached_pareto_front_mixin(problem: ArchOptTestProblemBase, discrete_problem: ArchOptTestProblemBase):
    problem.reset_pf_cache()
    assert not os.path.exists(problem._pf_cache_path())

    for _ in range(2):
        pf = problem.pareto_front(pop_size=200, n_gen_min=3, n_repeat=4)
        assert pf.shape[1] == 2
        assert os.path.exists(problem._pf_cache_path())

    discrete_problem.reset_pf_cache()
    assert not os.path.exists(discrete_problem._pf_cache_path())

    for _ in range(2):
        pf = discrete_problem.pareto_front()
        assert pf.shape == (10, 2)
        assert os.path.exists(discrete_problem._pf_cache_path())


def test_failing_evaluations(failing_problem: ArchOptTestProblemBase):
    out = failing_problem.evaluate(np.random.random((4, 5)), return_as_dictionary=True)
    is_failed = failing_problem.get_failed_points(out)
    assert np.all(is_failed == [True, False, True, False])
