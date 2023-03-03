import os
import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.pareto_front import *
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
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
        Integer(bounds=(0, 3)),
        Binary(),
        Choice(options=['A', 'B', 'C']),
    ])
    assert problem.n_var == 4
    assert problem.n_obj == 1

    assert np.all(problem.xl == [1, 0, 0, 0])
    assert np.all(problem.xu == [5, 3, 1, 2])
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
        [0, 7, 0, 0, 0],
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
            [0, 7, 0, 0, 0],
        ])
        assert np.all(repair.latest_is_active == [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, False],
        ])


def test_imputation_ratio(problem: ArchOptProblemBase, discrete_problem: ArchOptProblemBase):
    assert problem.get_n_declared_discrete() == 10*10
    assert problem.get_n_valid_discrete() == 10*10
    assert problem.get_imputation_ratio() == 1
    problem.print_stats()

    assert discrete_problem.get_n_declared_discrete() == 10*10
    assert discrete_problem.get_n_valid_discrete() == 10*5+5
    assert discrete_problem.get_imputation_ratio() == 1/.55
    discrete_problem.print_stats()


def test_evaluate(problem: ArchOptProblemBase):
    assert problem.n_var == 5

    assert problem.is_cat_mask[1]
    x_i = np.array([0, 1, 2, 3])
    cat_values = problem.get_categorical_values(1, x_i)
    assert np.all(cat_values == ['9', '8', '7', '6'])
    assert np.all((cat_values == '9') == (x_i == 0))

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
            [0, 7, 0, 0, 0],
        ])
        assert np.all(is_active == [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, False],
        ])
        assert np.all(f == [
            [0, 1],
            [0, 3.25],
            [0, 2.75],
        ])


def test_repaired_exhaustive_sampling(problem: ArchOptProblemBase):

    sampling_values = RepairedExhaustiveSampling.get_exhaustive_sample_values(problem)
    assert len(sampling_values) == 5
    assert np.prod([len(values) for values in sampling_values]) == 12500
    assert RepairedExhaustiveSampling.get_n_sample_exhaustive(problem) == 12500

    sampling = RepairedExhaustiveSampling()
    assert isinstance(sampling._repair, ArchOptRepair)
    x = sampling.do(problem, 100).get('X')
    assert x.shape == (7500, 5)
    assert np.unique(x, axis=0).shape[0] == 7500
    problem.evaluate(x)


def test_repaired_lhs_sampling(problem: ArchOptProblemBase):

    sampling = RepairedLatinHypercubeSampling()
    assert isinstance(sampling._repair, ArchOptRepair)
    x = sampling.do(problem, 1000).get('X')
    assert x.shape == (1000, 5)
    assert np.unique(x, axis=0).shape[0] == 1000
    problem.evaluate(x)

    init = get_init_sampler()
    x = init.do(problem, 1000).get('X')
    assert x.shape == (1000, 5)


def test_repaired_random_sampling(problem: ArchOptProblemBase):

    sampling_values = RepairedExhaustiveSampling.get_exhaustive_sample_values(problem)
    assert len(sampling_values) == 5
    assert np.prod([len(values) for values in sampling_values]) == 12500

    sampling = RepairedRandomSampling()
    assert isinstance(sampling._repair, ArchOptRepair)
    x = sampling.do(problem, 1000).get('X')
    assert x.shape == (1000, 5)
    assert np.unique(x, axis=0).shape[0] == 1000
    problem.evaluate(x)


def test_repaired_random_sampling_non_exhaustive(problem: ArchOptProblemBase):

    sampling_values = RepairedExhaustiveSampling.get_exhaustive_sample_values(problem)
    assert len(sampling_values) == 5
    assert np.prod([len(values) for values in sampling_values]) == 12500

    limit = RepairedRandomSampling._n_comb_gen_all_max
    RepairedRandomSampling._n_comb_gen_all_max = 10

    sampling = RepairedRandomSampling()
    assert isinstance(sampling._repair, ArchOptRepair)
    x = sampling.do(problem, 1000).get('X')
    assert x.shape == (1000, 5)
    assert np.unique(x, axis=0).shape[0] < 1000
    problem.evaluate(x)

    RepairedRandomSampling._n_comb_gen_all_max = limit


def test_cached_pareto_front_mixin(problem: ArchOptTestProblemBase, discrete_problem: ArchOptTestProblemBase):
    problem.reset_pf_cache()
    assert not os.path.exists(problem._pf_cache_path())

    for _ in range(2):
        pf = problem.pareto_front(pop_size=200, n_gen=3, n_repeat=4)
        assert pf.shape[1] == 2
        assert os.path.exists(problem._pf_cache_path())

    discrete_problem.reset_pf_cache()
    assert not os.path.exists(discrete_problem._pf_cache_path())

    for _ in range(2):
        pf = discrete_problem.pareto_front()
        assert pf.shape == (10, 2)
        assert os.path.exists(discrete_problem._pf_cache_path())
