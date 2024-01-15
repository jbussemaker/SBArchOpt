import itertools
import numpy as np
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.design_space import *
from pymoo.core.variable import Real, Integer, Binary, Choice


class DesignSpaceTest(ArchDesignSpace):

    def __init__(self, des_vars):
        self._des_vars = des_vars
        super().__init__()

    def is_explicit(self) -> bool:
        return False

    def _get_variables(self):
        return self._des_vars

    def _quick_sample_discrete_x(self, n: int):
        """Sample n discrete design vectors (also return is_active) without generating all design vectors first"""
        raise RuntimeError

    def _is_conditionally_active(self):
        return [False]*self.n_var

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        pass

    def _get_n_valid_discrete(self):
        pass

    def _get_n_active_cont_mean(self):
        pass

    def _get_n_correct_discrete(self):
        pass

    def _get_n_active_cont_mean_correct(self):
        pass

    def _gen_all_discrete_x(self):
        pass


def test_rounding():
    ds = DesignSpaceTest([Integer(bounds=(0, 5)), Integer(bounds=(-1, 1)), Integer(bounds=(2, 4))])

    assert np.all(ds.is_discrete_mask)
    x = np.array(list(itertools.product(np.linspace(0, 5, 20), np.linspace(-1, 1, 20), np.linspace(2, 4, 20))))
    ds.round_x_discrete(x)

    x1, x1_counts = np.unique(x[:, 0], return_counts=True)
    assert np.all(x1 == [0, 1, 2, 3, 4, 5])
    x1_counts = x1_counts / np.sum(x1_counts)
    assert np.all(np.abs(x1_counts - np.mean(x1_counts)) <= .05)

    x2, x2_counts = np.unique(x[:, 1], return_counts=True)
    assert np.all(x2 == [-1, 0, 1])
    x2_counts = x2_counts / np.sum(x2_counts)
    assert np.all(np.abs(x2_counts - np.mean(x2_counts)) <= .05)

    x3, x3_counts = np.unique(x[:, 2], return_counts=True)
    assert np.all(x3 == [2, 3, 4])
    x3_counts = x3_counts / np.sum(x3_counts)
    assert np.all(np.abs(x3_counts - np.mean(x3_counts)) <= .05)

    x_out_of_bounds = np.zeros((20, 3), dtype=int)
    x_out_of_bounds[:, 0] = np.linspace(-1, 6, 20)
    ds.round_x_discrete(x_out_of_bounds)
    assert np.all(np.min(x_out_of_bounds, axis=0) == [0, 0, 2])
    assert np.all(np.max(x_out_of_bounds, axis=0) == [5, 0, 2])


def test_init_no_vars():
    ds = DesignSpaceTest([])
    assert ds.n_var == 0
    assert ds.des_vars == []
    assert ds.get_n_declared_discrete() == 1
    assert np.isnan(ds.discrete_imputation_ratio)
    assert ds.continuous_imputation_ratio == 1.
    assert np.isnan(ds.imputation_ratio)
    assert np.isnan(ds.discrete_correction_ratio)
    assert ds.continuous_correction_ratio == 1.
    assert np.isnan(ds.correction_ratio)

    assert not ds.is_explicit()


def test_init_vars():
    ds = DesignSpaceTest([
        Real(bounds=(1, 5)),
        Integer(bounds=(1, 4)),
        Binary(),
        Choice(options=['A', 'B', 'C']),
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


def test_get_categorical_values():
    ds = DesignSpaceTest([Choice(options=['A', 'B', 'C'])])
    assert ds.all_discrete_x == (None, None)
    x_all, _ = ds.all_discrete_x_by_trial_and_imputation
    assert x_all.shape == (3, 1)

    x_all_, _ = ds.all_discrete_x
    assert np.all(x_all_ == x_all)

    cat_values = ds.get_categorical_values(x_all, 0)
    assert len(cat_values) == 3
    assert np.all(cat_values == ['A', 'B', 'C'])


def test_x_generation(problem: ArchOptProblemBase, discrete_problem: ArchOptProblemBase):
    for prob, n_valid, n_correct, cont_imp_ratio in [
        (problem, 10*10, 10*10, 1.2),
        (discrete_problem, 10*5+5, 10*10, 1.),
    ]:
        ds = prob.design_space
        assert ds.get_n_declared_discrete() == 10*10

        assert ds.get_n_valid_discrete() == n_valid
        assert ds.discrete_imputation_ratio == (10 * 10) / n_valid
        assert ds.continuous_imputation_ratio == cont_imp_ratio
        assert ds.imputation_ratio == cont_imp_ratio * (10*10)/n_valid

        assert ds.get_n_correct_discrete() == n_correct
        assert ds.discrete_correction_ratio == (10 * 10) / n_correct
        assert ds.continuous_correction_ratio == cont_imp_ratio
        assert ds.correction_ratio == cont_imp_ratio * (10*10)/n_correct

        assert ds.corrector is not None
        assert not ds.use_auto_corrector

        assert not ds.is_explicit()

        x_discrete, is_active_discrete = ds.all_discrete_x
        assert x_discrete.shape[0] == ds.get_n_valid_discrete()
        assert is_active_discrete.shape[0] == ds.get_n_valid_discrete()
        assert np.all(~LargeDuplicateElimination.eliminate(x_discrete))
        ds.get_discrete_rates(show=True)

        x_discrete_trial, is_act_trail = ds.all_discrete_x_by_trial_and_imputation
        assert np.all(x_discrete_trial == x_discrete)
        assert np.all(is_active_discrete == is_act_trail)

        np.random.seed(None)
        for _ in range(10):
            x_sampled, is_act_sampled = ds.quick_sample_discrete_x(20)
            assert x_sampled.shape == (20, ds.n_var)
            assert is_act_sampled.shape == (20, ds.n_var)

            x_sampled_, is_act_sampled_ = ds.correct_x(x_sampled)
            assert np.all(x_sampled_ == x_sampled)
            assert np.all(is_act_sampled_ == is_act_sampled)

        np.random.seed(42)
        x1, _ = ds.quick_sample_discrete_x(20)
        x2, _ = ds.quick_sample_discrete_x(20)
        assert np.any(x1 != x2)

        np.random.seed(42)
        x3, _ = ds.quick_sample_discrete_x(20)
        assert np.all(x1 == x3)
