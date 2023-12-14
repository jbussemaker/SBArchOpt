import os
import pytest
import itertools
import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.design_space import *
from sb_arch_opt.design_space_explicit import *
from sb_arch_opt.sampling import *
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from sb_arch_opt.problems.problems_base import *
from pymoo.core.variable import Real, Integer, Binary, Choice


def test_init_no_vars():
    problem = ArchOptProblemBase([], n_obj=2, n_ieq_constr=2)
    assert isinstance(problem.design_space, ArchDesignSpace)
    assert problem.n_var == 0
    assert problem.n_obj == 2
    assert problem.n_ieq_constr == 2

    assert problem.get_n_declared_discrete() == 1
    assert np.isnan(problem.get_imputation_ratio())
    problem.print_stats()

    with pytest.raises(RuntimeError):
        _ = problem.is_conditionally_active


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

    assert np.all(problem.is_conditionally_active == [False, False, False, False, True])


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
    assert problem.get_discrete_imputation_ratio() == 1
    assert problem.get_continuous_imputation_ratio() == 1.2
    assert problem.get_imputation_ratio() == 1.2
    problem.print_stats()

    x_discrete, is_active_discrete = problem.all_discrete_x
    assert x_discrete.shape[0] == problem.get_n_valid_discrete()
    assert is_active_discrete.shape[0] == problem.get_n_valid_discrete()
    assert np.all(~LargeDuplicateElimination.eliminate(x_discrete))
    problem.get_discrete_rates(show=True)

    x_discrete_trial, is_act_trail = problem.design_space.all_discrete_x_by_trial_and_imputation
    assert np.all(x_discrete_trial == x_discrete)
    assert np.all(is_active_discrete == is_act_trail)

    assert discrete_problem.get_n_declared_discrete() == 10*10
    assert discrete_problem.get_n_valid_discrete() == 10 * 5 + 5
    assert discrete_problem.get_discrete_imputation_ratio() == 1/.55
    assert discrete_problem.get_continuous_imputation_ratio() == 1.
    assert discrete_problem.get_imputation_ratio() == 1/.55
    discrete_problem.print_stats()

    x_discrete, is_active_discrete = discrete_problem.all_discrete_x
    assert x_discrete.shape[0] == discrete_problem.get_n_valid_discrete()
    assert is_active_discrete.shape[0] == discrete_problem.get_n_valid_discrete()
    assert np.all(~LargeDuplicateElimination.eliminate(x_discrete))
    discrete_problem.get_discrete_rates(show=True)

    x_discrete_trial, is_act_trail = discrete_problem.design_space.all_discrete_x_by_trial_and_imputation
    assert np.all(x_discrete_trial == x_discrete)
    assert np.all(is_active_discrete == is_act_trail)


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
            problem.get_population_statistics(pop, show=True)
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

    problem.get_population_statistics(Population.new(), show=True)


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
    is_dup = LargeDuplicateElimination.eliminate(x)
    assert np.all(np.where(~is_dup)[0] == [0, 1, 3, 4, 6])
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


def test_hierarchical_exhaustive_sampling(problem: ArchOptProblemBase):
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

    def __init__(self, n=4, has_cont=True):
        self._n = n
        des_vars = []
        for _ in range(n):
            des_vars.append(Binary())
        if has_cont:
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


def test_hierarchical_exhaustive_sampling_hierarchical():
    problem = HierarchicalDummyProblem()
    x = HierarchicalExhaustiveSampling(n_cont=2).do(problem, 0).get('X')
    assert x.shape == (31, 8)  # 2**(n+1)-1

    x_imp, _ = problem.correct_x(x)
    assert np.all(x_imp == x)


def test_hierarchical_exhaustive_sampling_hierarchical_large():
    n = 12
    problem = HierarchicalDummyProblem(n=n)
    x = HierarchicalExhaustiveSampling(n_cont=2).do(problem, 0).get('X')
    assert x.shape == (2**(n+1)-1, n*2)

    x_imp, _ = problem.correct_x(x)
    assert np.all(x_imp == x)


def test_sobol_sampling():
    for _ in range(100):
        i = HierarchicalSampling._sobol_choice(5, 10, replace=True)
        assert len(i) == 5
        assert 0 <= np.min(i) <= np.max(i) < 10
        assert len(np.unique(i)) <= 10

        i = HierarchicalSampling._sobol_choice(10, 5, replace=True)
        assert len(i) == 10
        assert 0 <= np.min(i) <= np.max(i) < 5
        assert len(np.unique(i)) <= 5

        i = HierarchicalSampling._sobol_choice(5, 10, replace=False)
        assert len(i) == 5
        assert 0 <= np.min(i) <= np.max(i) < 10
        assert len(np.unique(i)) == len(i)

    with pytest.raises(ValueError):
        HierarchicalSampling._sobol_choice(10, 5, replace=False)

    np.random.seed(42)
    i1 = HierarchicalSampling._sobol_choice(5, 100)
    i2 = HierarchicalSampling._sobol_choice(5, 100)
    assert np.any(i1 != i2)

    np.random.seed(42)
    i3 = HierarchicalSampling._sobol_choice(5, 100)
    assert np.all(i1 == i3)


def _disable_x_all(problem: ArchOptProblemBase):
    def _raise():
        raise MemoryError

    problem.design_space.all_discrete_x = (None, None)
    problem.design_space._get_all_discrete_x_by_trial_and_imputation = _raise


def test_hierarchical_random_sampling(problem: ArchOptProblemBase):

    sampling_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(problem)
    assert len(sampling_values) == 5
    assert np.prod([len(values) for values in sampling_values]) == 12500

    _disable_x_all(problem)

    for sobol in [False, True]:
        sampling = HierarchicalSampling(sobol=sobol)
        assert isinstance(sampling._repair, ArchOptRepair)
        assert sampling.get_hierarchical_cartesian_product(problem, sampling.repair) == (None, None)

        x = sampling.do(problem, 1000).get('X')
        assert x.shape == (1000, 5)
        assert np.unique(x, axis=0).shape[0] == 1000
        problem.evaluate(x)

        x_imp, _ = problem.correct_x(x)
        assert np.all(x_imp == x)

        np.random.seed(42)
        x1 = sampling.do(problem, 1000).get('X')
        x2 = sampling.do(problem, 1000).get('X')
        assert np.any(x1 != x2)

        np.random.seed(42)
        x3 = sampling.do(problem, 1000).get('X')
        assert np.all(x1 == x3)

        x4 = HierarchicalSampling(sobol=sobol, seed=42).do(problem, 1000).get('X')
        assert np.all(x1 == x4)


def test_hierarchical_random_sampling_discrete_hierarchical(discrete_problem: ArchOptProblemBase):

    sampling_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(discrete_problem)
    assert len(sampling_values) == 2
    assert np.prod([len(values) for values in sampling_values]) == 100

    _disable_x_all(discrete_problem)

    for sobol in [False, True]:
        sampling = HierarchicalSampling(sobol=sobol)
        assert isinstance(sampling._repair, ArchOptRepair)
        assert sampling.get_hierarchical_cartesian_product(discrete_problem, sampling.repair) == (None, None)

        x = sampling.do(discrete_problem, 1000).get('X')
        assert x.shape == (55, 2)
        assert np.unique(x, axis=0).shape[0] == 55
        discrete_problem.evaluate(x)

        x_imp, _ = discrete_problem.correct_x(x)
        assert np.all(x_imp == x)

    for sobol in [False, True]:
        sampling = HierarchicalSampling(sobol=sobol)
        assert isinstance(sampling._repair, ArchOptRepair)
        assert sampling.get_hierarchical_cartesian_product(discrete_problem, sampling.repair) == (None, None)

        x = sampling.do(discrete_problem, 50).get('X')
        assert x.shape[0] < 50
        assert np.unique(x, axis=0).shape[0] == x.shape[0]
        discrete_problem.evaluate(x)

        x_imp, _ = discrete_problem.correct_x(x)
        assert np.all(x_imp == x)


def test_hierarchical_random_sampling_non_exhaustive(problem: ArchOptProblemBase):

    sampling_values = HierarchicalExhaustiveSampling.get_exhaustive_sample_values(problem)
    assert len(sampling_values) == 5
    assert np.prod([len(values) for values in sampling_values]) == 12500

    _disable_x_all(problem)

    for sobol in [False, True]:
        sampling = HierarchicalSampling(sobol=sobol)
        assert isinstance(sampling._repair, ArchOptRepair)
        assert sampling.get_hierarchical_cartesian_product(problem, sampling.repair) == (None, None)

        x = sampling.do(problem, 1000).get('X')
        assert x.shape == (1000, 5)
        assert np.unique(x, axis=0).shape[0] == 1000
        problem.evaluate(x)

        x_imp, _ = problem.correct_x(x)
        assert np.all(x_imp == x)

        np.random.seed(42)
        x1 = sampling.do(problem, 1000).get('X')
        x2 = sampling.do(problem, 1000).get('X')
        assert np.any(x1 != x2)

        np.random.seed(42)
        x3 = sampling.do(problem, 1000).get('X')
        assert np.all(x1 == x3)


def test_cached_pareto_front_mixin(problem: ArchOptTestProblemBase, discrete_problem: ArchOptTestProblemBase):
    problem.reset_pf_cache()
    assert not os.path.exists(problem._pf_cache_path())

    for _ in range(2):
        pf = problem.pareto_front(pop_size=20, n_gen_min=3, n_repeat=4)
        assert pf.shape[1] == 2
        assert os.path.exists(problem._pf_cache_path())

    problem.reset_pf_cache()
    assert not os.path.exists(problem._pf_cache_path())

    for _ in range(2):
        assert problem.pareto_set(pop_size=20, n_gen_min=3, n_repeat=4) is not None
        assert os.path.exists(problem._pf_cache_path())

    ps, pf = problem.pareto_set(), problem.pareto_front()
    assert ps.shape[0] == pf.shape[0]
    out = problem.evaluate(ps, return_as_dictionary=True)
    assert np.all(np.abs(out['X']-ps) < 1e-10)
    assert np.all(np.abs(out['F']-pf) < 1e-10)

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


class DummyExplicitDesignSpaceProblem(ArchOptTestProblemBase):

    def __init__(self, write_x_out=False):
        ds = ExplicitArchDesignSpace([
            CategoricalParam('a', ['A', 'B', 'C']),
            IntegerParam('b', 0, 3),
            ContinuousParam('c', 0, 2),
        ])
        ds.add_conditions([
            InCondition(ds['b'], ds['a'], ['A', 'B']),
            InCondition(ds['c'], ds['a'], ['B', 'C']),
        ])
        ds.add_value_constraint(ds['b'], 3, ds['a'], 'B')
        self.write_x_out = write_x_out
        super().__init__(ds)

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        for i, xi in enumerate(x):
            assert is_active_out[i, 1] == (xi[0] != 2)
            assert is_active_out[i, 2] == (xi[0] != 0)

            if xi[0] == 0:  # A
                assert xi[2] == 1.  # c is inactive

            elif xi[0] == 2:  # B
                assert xi[1] != 3  # b == 3 forbidden if a == B

        f_out[:, :] = 0

        if self.write_x_out:
            x[:, :] += 1
            is_active_out[:, :] = False

    def __repr__(self):
        return f'{self.__class__.__name__}'


def test_explicit_design_space():
    problem = DummyExplicitDesignSpaceProblem()

    assert problem.get_n_valid_discrete() == 8
    x_all, _ = problem.all_discrete_x
    assert x_all.shape == (8, 3)
    problem.evaluate(x_all)

    problem_try_mod = DummyExplicitDesignSpaceProblem(write_x_out=True)
    with pytest.raises(ValueError):
        problem_try_mod.evaluate(x_all)
