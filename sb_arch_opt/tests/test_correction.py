import numpy as np
from typing import *
from pymoo.core.variable import Real, Choice
from sb_arch_opt.design_space import ArchDesignSpace
from sb_arch_opt.correction import *


class DummyArchDesignSpace(ArchDesignSpace):

    def __init__(self, x_all: np.ndarray, is_act_all: np.ndarray, is_discrete_mask: np.ndarray = None):
        self._corrector = None
        self._x_all = x_all
        self._is_act_all = is_act_all

        if is_discrete_mask is None:
            is_discrete_mask = np.ones((x_all.shape[1],), dtype=bool)
        self._is_discrete_mask = is_discrete_mask
        super().__init__()

        self.use_auto_corrector = True

    @property
    def corrector(self):
        if self._corrector is None:
            self._corrector = self._get_corrector()
        return self._corrector

    @corrector.setter
    def corrector(self, corrector):
        self._corrector = corrector

    def is_explicit(self) -> bool:
        return False

    def _get_variables(self):
        des_vars = []
        for i, is_discrete in enumerate(self._is_discrete_mask):
            if is_discrete:
                des_vars.append(Choice(options=list(sorted(np.unique(self._x_all[:, i])))))
            else:
                des_vars.append(Real(bounds=(0., 1.)))
        return des_vars

    def _is_conditionally_active(self) -> Optional[List[bool]]:
        pass  # Derived from is_active_all

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        raise RuntimeError

    def _quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        raise RuntimeError

    def _get_n_valid_discrete(self) -> Optional[int]:
        return self._x_all.shape[0]

    def _get_n_active_cont_mean(self) -> Optional[float]:
        pass

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._x_all, self._is_act_all

    def is_valid(self, xi: np.ndarray) -> Optional[np.ndarray]:
        eager_corr = EagerCorrectorBase(self)
        i_valid = eager_corr.get_correct_idx(np.array([xi]))[0]
        if i_valid == -1:
            return
        _, is_active_valid = eager_corr.x_valid_active
        return is_active_valid[i_valid, :]


def test_corrector():
    x_all = np.array([[0, 0],
                      [1, 0],
                      [0, 1],
                      [1, 1]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)
    assert np.all(ds.is_discrete_mask)

    corr = CorrectorBase(ds)
    assert np.all(corr.is_discrete_mask)
    assert np.all(corr.x_imp_discrete == np.array([0, 0]))

    assert corr._is_canonical_inactive(np.array([1, 1]), np.array([True, True]))
    assert corr._is_canonical_inactive(np.array([1, 0]), np.array([True, False]))
    assert not corr._is_canonical_inactive(np.array([1, 1]), np.array([True, False]))


def test_eager_corrector():
    x_all = np.array([[0, 0],
                      [0, 1],
                      [0, 2],
                      [1, 0],
                      [1, 1],
                      [2, 0]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-1, 1] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    corr = EagerCorrectorBase(ds)
    x_try = np.array([[1, 0],  # Canonical, 3
                      [1, 2],  # Invalid
                      [2, 0],  # Canonical, 5
                      [2, 1]])  # Valid, non-canonical, 5

    assert np.all(corr.get_correct_idx(x_try) == np.array([3, -1, 5, 5]))
    assert np.all(corr.get_canonical_idx(x_try) == np.array([3, -1, 5, -1]))


def test_closest_eager_corrector():
    x_all = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 1],
                      [0, 1, 2],
                      [1, 0, 0],
                      [1, 0, 2],
                      [1, 1, 0],
                      [2, 0, 0],
                      [2, 1, 3]])
    is_act_all = np.ones(x_all.shape, dtype=bool)
    is_act_all[-3, 2] = False
    is_act_all[-2, 1:] = False
    ds = DummyArchDesignSpace(x_all=x_all, is_act_all=is_act_all)

    for correct_correct_x in [False, True]:
        for random_if_multiple in [False, True]:
            for _ in range(10 if random_if_multiple else 1):
                for euclidean in [False, True]:
                    ds.corrector = corr = ClosestEagerCorrector(
                        ds, euclidean=euclidean, correct_correct_x=correct_correct_x,
                        random_if_multiple=random_if_multiple)
                    assert repr(corr)

                    x_corr, is_act_corr = ds.correct_x(np.array([[0, 0, 0],
                                                                 [0, 0, 3],
                                                                 [1, 0, 1],
                                                                 [1, 1, 1],
                                                                 [1, 1, 2],
                                                                 [2, 0, 2]]))

                    x_corr_, is_act_corr_ = ds.correct_x(x_corr)
                    assert np.all(x_corr == x_corr_)
                    assert np.all(is_act_corr == is_act_corr_)

                    corr_first = np.array([[0, 0, 0],
                                           [0, 0, 1],
                                           [1, 0, 0],
                                           [1, 1, 0],
                                           [1, 1, 0],
                                           [2, 0, 0]])
                    if euclidean:
                        corr_first[1, :] = [0, 1, 2]
                    if correct_correct_x:
                        corr_first[-2, :] = [1, 0, 2]
                        corr_first[-1, :] = [1, 0, 2]
                    corr_second = corr_first.copy()
                    corr_second[2, :] = [1, 0, 2]

                    if random_if_multiple:
                        assert np.all(x_corr == corr_first) or np.all(x_corr == corr_second)
                    else:
                        assert np.all(x_corr == corr_first)
