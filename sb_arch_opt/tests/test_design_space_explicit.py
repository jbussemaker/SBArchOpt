import pytest
import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.design_space_explicit import *
from pymoo.operators.sampling.rnd import FloatRandomSampling


def test_init_no_vars():
    ds = ExplicitArchDesignSpace()
    assert ds.n_var == 0
    assert ds.des_vars == []
    assert ds.get_n_declared_discrete() == 1
    assert ds.discrete_imputation_ratio == 1.
    assert ds.continuous_imputation_ratio == 1
    assert ds.imputation_ratio == 1
    assert ds.discrete_correction_ratio == 1.
    assert ds.continuous_correction_ratio == 1
    assert ds.correction_ratio == 1

    assert ds.is_explicit()

    with pytest.raises(RuntimeError):
        ds.add_param(ContinuousParam('a', 0, 6))


def test_init_vars():
    ds = ExplicitArchDesignSpace([
        ContinuousParam('a', 1., 5.),
        IntegerParam('b', 1, 4),
        IntegerParam('c', 0, 1),
        CategoricalParam('d', options=['A', 'B', 'C']),
    ])
    assert ds.n_var == 4

    assert np.all(ds.xl == [1, 1, 0, 0])
    assert np.all(ds.xu == [5, 4, 1, 2])
    assert np.all(ds.is_cat_mask == [False, False, False, True])
    assert np.all(ds.is_int_mask == [False, True, True, False])
    assert np.all(ds.is_discrete_mask == [False, True, True, True])
    assert np.all(ds.is_cont_mask == [True, False, False, False])
    assert np.all(~ds.is_conditionally_active)

    assert ds.get_n_declared_discrete() == 4*2*3

    assert ds.discrete_imputation_ratio == 1
    assert ds.continuous_imputation_ratio == 1
    assert ds.imputation_ratio == 1

    assert ds.discrete_correction_ratio == 1
    assert ds.continuous_correction_ratio == 1
    assert ds.correction_ratio == 1


def test_num_x():
    ds = ExplicitArchDesignSpace([
        ContinuousParam('a', 0, 1),
        ContinuousParam('b', -1, 6),
        IntegerParam('c', 0, 1),
        IntegerParam('d', -1, 6),
        IntegerParam('e', 0, 3),
    ])

    x, is_active = ds.all_discrete_x
    assert x.shape == (64, 5)
    assert np.all(x[:, :2] == [0, -1])
    assert np.all(np.min(x[:, 2:], axis=0) == [0, -1, 0])
    assert np.all(np.max(x[:, 2:], axis=0) == [1, 6, 3])
    assert len(np.unique(x, axis=0)) == x.shape[0]
    assert is_active.shape == x.shape
    assert np.all(is_active)

    assert ds.get_n_declared_discrete() == 64
    assert ds.get_n_valid_discrete() == 64
    assert ds.discrete_imputation_ratio == 1
    assert ds.continuous_imputation_ratio == 1
    assert np.all(~ds.is_conditionally_active)

    x, is_active = ds.quick_sample_discrete_x(1000)
    assert x.shape == (1000, 5)
    assert np.all(np.abs(np.min(x, axis=0)-ds.xl) < .1)
    assert np.all(np.abs(np.max(x, axis=0)-ds.xu) < .1)
    assert is_active.shape == x.shape
    assert np.all(is_active)

    x_imp, is_active_imp = ds.correct_x(x)
    assert np.all(x_imp == x)
    assert np.all(is_active_imp)

    ds.correct_x(np.array([[1+1e-4, 0, 0, 0, 0]]))


def test_discrete_x():
    ds = ExplicitArchDesignSpace([
        IntegerParam('a', 0, 1),
        CategoricalParam('b', options=['A', 'B', 'C']),
    ])
    assert ds.n_var == 2

    x, is_active = ds.all_discrete_x
    assert x.shape == (6, 2)
    assert np.all(np.min(x, axis=0) == [0, 0])
    assert np.all(np.max(x, axis=0) == [1, 2])
    assert len(np.unique(x, axis=0)) == x.shape[0]
    assert is_active.shape == x.shape
    assert np.all(is_active)

    assert ds.get_n_declared_discrete() == 6
    assert ds.get_n_valid_discrete() == 6
    assert ds.discrete_imputation_ratio == 1
    assert ds.continuous_imputation_ratio == 1
    assert np.all(~ds.is_conditionally_active)

    x, is_active = ds.quick_sample_discrete_x(100)
    assert x.shape == (100, 2)
    assert np.all(np.min(x, axis=0) == [0, 0])
    assert np.all(np.max(x, axis=0) == [1, 2])
    assert len(np.unique(x, axis=0)) == 6
    assert is_active.shape == x.shape
    assert np.all(is_active)

    x_imp, is_active_imp = ds.correct_x(x)
    assert np.all(x_imp == x)
    assert np.all(is_active_imp)

    x, is_active = HierarchicalSampling(sobol=False).randomly_sample(
        ArchOptProblemBase(ds), 100, ArchOptRepair(), x_all=None, is_act_all=None)
    assert x.shape == (6, 2)
    assert is_active.shape == x.shape
    assert np.all(is_active)

    np.random.seed(42)
    x1, _ = ds.quick_sample_discrete_x(20)
    x2, _ = ds.quick_sample_discrete_x(20)
    assert np.any(x1 != x2)

    np.random.seed(42)
    x3, _ = ds.quick_sample_discrete_x(20)
    assert np.all(x1 == x3)


def test_hierarchy():
    ds = ExplicitArchDesignSpace([
        CategoricalParam('a', options=['A', 'B', 'C']),
        CategoricalParam('b', options=['E', 'F']),
        IntegerParam('c', 0, 1),
        ContinuousParam('d', 0, 1),
    ])
    ds.add_conditions([
        EqualsCondition(ds['b'], ds['a'], 'A'),  # Activate b if a == A
        EqualsCondition(ds['c'], ds['a'], 'B'),  # Activate c if a == B
        InCondition(ds['d'], ds['a'], ['B', 'C']),  # Activate d if a == B or C
    ])

    assert ds.n_var == 4
    assert ds.get_n_declared_discrete() == 12
    assert np.all(ds.is_conditionally_active == [False, True, True, True])

    x, is_active = ds.all_discrete_x
    assert x.shape == (5, 4)
    assert is_active.shape == x.shape
    assert np.all(is_active == [
        [True, True, False, False],
        [True, True, False, False],
        [True, False, True, True],
        [True, False, True, True],
        [True, False, False, True],
    ])

    assert ds.get_n_valid_discrete() == 5
    assert ds.discrete_imputation_ratio == 12 / 5
    assert ds.continuous_imputation_ratio == 1/(3/5)
    assert ds.imputation_ratio == (12/5) * (5/3)

    assert np.all(ds.all_discrete_x_n_correct == [2, 2, 2, 2, 4])
    assert ds.get_n_correct_discrete() == 12
    assert ds.discrete_correction_ratio == 12 / 12
    assert ds.continuous_correction_ratio == 12 / (12-4)
    assert ds.correction_ratio == (12/12) * (3/2)

    x, is_active = ds.quick_sample_discrete_x(100)
    assert x.shape == (100, 4)

    x, is_active = HierarchicalSampling(sobol=False).randomly_sample(
        ArchOptProblemBase(ds), 100, ArchOptRepair(), x_all=None, is_act_all=None)
    assert x.shape == (100, 4)
    assert is_active.shape == x.shape
    _, is_active_ = ds.correct_x(x)
    assert np.all(is_active == is_active_)

    x = HierarchicalSampling().do(ArchOptProblemBase(ds), 100).get('X')
    assert x.shape == (100, 4)

    x_random = FloatRandomSampling().do(ArchOptProblemBase(ds), 100).get('X')
    assert x_random.shape == (100, 4)
    x, is_active = ds.correct_x(x_random)
    assert x.shape == (100, 4)
    assert not np.all(is_active)

    x_, is_active_ = ds.correct_x(x)
    assert np.all(x_ == x)
    assert np.all(is_active_ == is_active)


def test_sorting():
    ds = ExplicitArchDesignSpace([
        IntegerParam('a', 0, 1),
        CategoricalParam('b', options=['A', 'B', 'C']),
    ])
    ds.add_conditions([
        EqualsCondition(ds['a'], ds['b'], 'A'),  # Activate a if b == A
    ])

    assert ds.n_var == 2
    assert ds.get_param_names() == ['a', 'b']
    assert ds.config_space.get_hyperparameter_names() == ['b', 'a']
    assert ds.get_params_dict() == {
        'a': ds['a'],
        'b': ds['b'],
    }
    assert ds.get_param_by_idx(0) == 'a'
    assert ds.get_param_by_idx(1) == 'b'
    assert ds.get_idx_by_param_name('a') == 0
    assert ds.get_idx_by_param_name('b') == 1
    assert np.all(ds.is_conditionally_active == [True, False])

    x, is_active = ds.all_discrete_x
    assert x.shape == (4, 2)
    assert np.all(x == [
        [0, 1],
        [0, 2],
        [0, 0],
        [1, 0],
    ])
    assert np.all(is_active == [
        [False, True],
        [False, True],
        [True, True],
        [True, True],
    ])

    x_imp, is_active_imp = ds.correct_x(x)
    assert np.all(x == x_imp)
    assert np.all(is_active == is_active_imp)

    x, is_active = ds.quick_sample_discrete_x(100)
    assert np.all(np.min(x, axis=0) == [0, 0])
    assert np.all(np.max(x, axis=0) == [1, 2])


def test_forbidden():
    ds = ExplicitArchDesignSpace([
        CategoricalParam('a', options=['A', 'B', 'C']),
        CategoricalParam('b', options=['E', 'F']),
        IntegerParam('c', 0, 1),
        ContinuousParam('d', 0, 1),
    ])
    ds.add_conditions([
        EqualsCondition(ds['d'], ds['a'], 'A'),  # Activate d if a == A
    ])
    ds.add_forbidden_clause(ForbiddenAndConjunction(  # Forbid b == F iff a == C
        ForbiddenEqualsClause(ds['a'], 'C'), ForbiddenEqualsClause(ds['b'], 'F'),
    ))

    assert ds.n_var == 4
    assert ds.get_n_declared_discrete() == 12
    assert np.all(ds.is_conditionally_active == [False, False, False, True])

    x, is_active = ds.quick_sample_discrete_x(100)
    assert x.shape == (100, 4)

    x_, is_active_ = ds.correct_x(x)
    assert np.all(x_ == x)
    assert np.all(is_active_ == is_active)

    x, is_active = ds.all_discrete_x
    assert x.shape == (10, 4)
    assert is_active.shape == x.shape
    assert np.all(is_active == [
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
    ])

    assert ds.discrete_imputation_ratio == 1.2
    assert ds.continuous_imputation_ratio == 10/4
    assert ds.imputation_ratio == (10/4) * 1.2

    assert np.all(ds.all_discrete_x_n_correct == 1)
    assert ds.discrete_correction_ratio == 1.2
    assert ds.continuous_correction_ratio == 10/4
    assert ds.correction_ratio == (10/4) * 1.2

    x = HierarchicalSampling().do(ArchOptProblemBase(ds), 100).get('X')
    assert x.shape == (100, 4)

    x_random = FloatRandomSampling().do(ArchOptProblemBase(ds), 100).get('X')
    assert x_random.shape == (100, 4)
    x, is_active = ds.correct_x(x_random)
    assert x.shape == (100, 4)
    assert not np.all(is_active)

    x_, is_active_ = ds.correct_x(x)
    assert np.all(x_ == x)
    assert np.all(is_active_ == is_active)


def test_add_value_constraint():
    for src_as_list, tgt_as_list in [(False, False), (True, False), (False, True), (True, True)]:
        ds = ExplicitArchDesignSpace([
            CategoricalParam('a', options=['A', 'B', 'C']),
            CategoricalParam('b', options=['E', 'F']),
            IntegerParam('c', 0, 1),
            ContinuousParam('d', 0, 1),
        ])
        ds.add_conditions([
            EqualsCondition(ds['d'], ds['a'], 'A'),  # Activate d if a == A
        ])
        # Forbid b == F if a == C
        ds.add_value_constraint(ds['b'], ['F'] if tgt_as_list else 'F', ds['a'], ['C'] if src_as_list else 'C')

        assert ds.n_var == 4
        assert ds.get_n_declared_discrete() == 12
        assert np.all(ds.is_conditionally_active == [False, False, False, True])

        x, is_active = ds.all_discrete_x
        assert x.shape == (10, 4)


def test_add_value_constraint_default_config():
    ds = ExplicitArchDesignSpace([
        IntegerParam('a', 0, 2),
        IntegerParam('b', 0, 2),
    ])
    assert ds.config_space.get_hyperparameters()[0].default_value == 1

    ds.add_value_constraint(ds['a'], 1, ds['b'], 1)
    ds.quick_sample_discrete_x(100)

    x_random = FloatRandomSampling().do(ArchOptProblemBase(ds), 100).get('X')
    assert x_random.shape == (100, 2)
    ds.correct_x(x_random)


@pytest.mark.skip('Cyclic conditions not (yet?) supported by ConfigSpace')
def test_circular_conditions():
    ds = ExplicitArchDesignSpace([
        CategoricalParam('x1', options=[0, 1]),
        CategoricalParam('x2', options=[0, 1]),
        CategoricalParam('x3', options=[0, 1]),
    ])
    ds.add_conditions([
        OrConjunction(  # x2 is active if x1 == 0 OR (x1 == 1 and x3 == 0)
            EqualsCondition(ds['x2'], ds['x1'], 0),
            AndConjunction(
                EqualsCondition(ds['x2'], ds['x1'], 1),
                EqualsCondition(ds['x2'], ds['x3'], 0),
            ),
        ),
        OrConjunction(  # x3 is active if x1 == 1 OR (x1 == 0 and x2 == 1)
            EqualsCondition(ds['x3'], ds['x1'], 1),
            AndConjunction(
                EqualsCondition(ds['x3'], ds['x1'], 0),
                EqualsCondition(ds['x3'], ds['x2'], 1),
            ),
        ),
    ])

    x_all, is_act_all = ds.all_discrete_x
    assert x_all.shape == (6, 3)
    assert np.all(x_all == [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
    ])
    assert is_act_all.shape == x_all.shape
    assert np.all(is_act_all == [
        [True, True, False],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, False, True],
    ])

    assert ds.get_n_valid_discrete() == 6
    assert np.all(ds.is_conditionally_active == [False, True, True])

    x, is_active = ds.quick_sample_discrete_x(100)
    assert len(np.unique(x, axis=0)) == 6
    assert len(np.unique(is_active, axis=0)) == 3

    for i, xi in enumerate(x):
        x_imp, is_act_imp = ds.correct_x(xi)
        assert np.all(xi == x_imp)
        assert np.all(is_active[i, :] == is_act_imp)

    x_random = FloatRandomSampling().do(ArchOptProblemBase(ds), 100)
    x, is_active = ds.correct_x(x_random)
    assert len(np.unique(x, axis=0)) == 6
    assert len(np.unique(is_active, axis=0)) == 3
