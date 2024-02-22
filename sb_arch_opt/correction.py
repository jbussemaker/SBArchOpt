"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from typing import Optional, Tuple
from scipy.spatial import distance
from cached_property import cached_property

from sb_arch_opt.design_space import ArchDesignSpace, CorrectorInterface, CorrectorUnavailableError

__all__ = ['CorrectorBase', 'EagerCorrectorBase', 'ClosestEagerCorrector']


class CorrectorBase(CorrectorInterface):
    """
    Base class implementing some generic correction algorithm.
    Correction is the mechanism of taking any input design vector x and ensuring it is a correct design vector, that is:
    all (hierarchical) value constraints are satisfied.
    Imputation is the mechanism of turning a correct vector into a valid vector, that is, an x where inactive
    variables are replaced by canonical values: 0 (discrete) or mid-bounds (continuous).

    We assume that only discrete variables determine activeness and are subject to value constraints, so only
    discrete variables need to be corrected.

    From this, there are three "statuses" that design vectors can have:
    - Valid: correct, and inactive discrete variables are imputed (canonical)
    - Correct: active discrete variables represent a correct combination (all value constraints are satisfied)
    - Invalid: one or more value constraints are violated (for discrete variables)

    Invalid design vectors always need to be corrected to a correct/valid design vector.
    Correct design vectors may optionally be "corrected" to a valid design vector too, which allows non-canonical
    design vectors to be modified.
    """

    default_correct_correct_x = False

    def __init__(self, design_space: ArchDesignSpace, correct_correct_x: bool = None):
        self._design_space = design_space
        self.correct_correct_x = self.default_correct_correct_x if correct_correct_x is None else correct_correct_x

    @property
    def design_space(self) -> ArchDesignSpace:
        """Mask specifying for each design variable whether it is a discrete variable or not."""
        return self._design_space

    @cached_property
    def is_discrete_mask(self) -> np.ndarray:
        return self._design_space.is_discrete_mask

    @cached_property
    def is_cont_mask(self) -> np.ndarray:
        return self._design_space.is_cont_mask

    @cached_property
    def x_imp_discrete(self) -> np.ndarray:
        return self._design_space.xl[self._design_space.is_discrete_mask]

    def _is_canonical_inactive(self, xi: np.ndarray, is_active_i: np.ndarray) -> bool:
        # Check whether each discrete variable has its corresponding imputed value
        is_discrete_mask = self.is_discrete_mask
        is_x_imp_discrete = xi[is_discrete_mask] == self.x_imp_discrete

        # Check which discrete design variables are inactive
        is_inactive = ~is_active_i[is_discrete_mask]

        # Check if all inactive discrete design variables have their corresponding imputed values
        return np.all(is_x_imp_discrete[is_inactive])

    def correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        No need to impute inactive design variables.
        """
        # Quit if there are no discrete design variables
        if not np.any(self.is_discrete_mask):
            return

        is_cont_mask = self.is_cont_mask
        x_in_cont = x[:, is_cont_mask].copy()

        # Correct discrete variables
        self._correct_x(x, is_active)

        # Retain values of continuous variables
        x[:, is_cont_mask] = x_in_cont

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        No need to impute inactive design variables.
        """
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def __repr__(self):
        raise NotImplementedError


class EagerCorrectorBase(CorrectorBase):
    """
    Corrector that has access to the list of all valid discrete design vectors.
    """

    default_random_if_multiple = False

    def __init__(self, design_space: ArchDesignSpace, correct_correct_x: bool = None, random_if_multiple: bool = None):
        self._x_valid = None
        self._random_if_multiple = self.default_random_if_multiple if random_if_multiple is None else random_if_multiple
        super().__init__(design_space, correct_correct_x=correct_correct_x)

    @property
    def x_valid_active(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._design_space.all_discrete_x

    @cached_property
    def _x_canonical_map(self) -> dict:
        x_canonical_map = {}
        is_discrete_mask = self.is_discrete_mask
        x_valid, _ = self.x_valid_active
        for i, xi in enumerate(x_valid):
            x_canonical_map[tuple(xi[is_discrete_mask])] = i
        return x_canonical_map

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        x_valid, is_active_valid = self.x_valid_active
        if x_valid is None or is_active_valid is None:
            raise CorrectorUnavailableError('Eager corrector unavailable because problem does not provide x_all')

        # Separate canonical design vectors
        correct_idx = self.get_canonical_idx(x) if self.correct_correct_x else self.get_correct_idx(x)
        is_correct = correct_idx != -1
        to_be_corrected = ~is_correct

        # Set activeness information of correct vectors
        is_active[is_correct, :] = is_active_valid[correct_idx[is_correct], :]

        # Check if anything should be corrected
        if not np.any(to_be_corrected):
            return

        # Get corrected design vector indices
        xi_corrected = self._get_corrected_x_idx(x[to_be_corrected, :])
        if len(xi_corrected.shape) != 1:
            raise ValueError(f'Expecting vector of length {x[to_be_corrected].shape[0]}, got {xi_corrected.shape}')

        # Correct design vectors and return activeness information
        x[to_be_corrected, :] = x_valid[xi_corrected, :]
        is_active[to_be_corrected, :] = is_active_valid[xi_corrected, :]

    def get_canonical_idx(self, x: np.ndarray) -> np.ndarray:
        """Returns a vector specifying for each vector the corresponding valid design vector if the vector is also
        canonical or -1 if not the case."""

        x_canonical_map = self._x_canonical_map
        is_discrete_mask = self.is_discrete_mask

        canonical_idx = -np.ones(x.shape[0], dtype=int)
        for i, xi in enumerate(x):
            ix_canonical = x_canonical_map.get(tuple(xi[is_discrete_mask]))
            if ix_canonical is not None:
                canonical_idx[i] = ix_canonical

        return canonical_idx

    def get_correct_idx(self, x: np.ndarray) -> np.ndarray:
        """Returns a vector specifying for each vector the corresp. valid design vector idx or -1 if not found."""
        valid_idx = -np.ones(x.shape[0], dtype=int)
        for i, xi in enumerate(x):
            ix_valid = self._get_valid_idx_single(xi)
            if ix_valid is not None:
                valid_idx[i] = ix_valid
        return valid_idx

    def _get_valid_idx_single(self, xi: np.ndarray) -> Optional[int]:
        """Returns a valid design vector index for a given design vector, or None if not found"""
        is_discrete_mask = self.is_discrete_mask

        # Check if vector is canonical
        x_canonical_map = self._x_canonical_map
        ix_canonical = x_canonical_map.get(tuple(xi[is_discrete_mask]))
        if ix_canonical is not None:
            return ix_canonical

        x_valid, is_active_valid = self.x_valid_active
        matched_dv_idx = np.arange(x_valid.shape[0])
        x_valid_matched, is_active_valid_matched = x_valid, is_active_valid
        for i, is_discrete in enumerate(is_discrete_mask):
            # Ignore continuous vars
            if not is_discrete:
                continue

            # Match active valid x to value or inactive valid x
            is_active_valid_i = is_active_valid_matched[:, i]
            matched = (is_active_valid_i & (x_valid_matched[:, i] == xi[i])) | (~is_active_valid_i)

            # Select vectors and check if there are any vectors left to choose from
            matched_dv_idx = matched_dv_idx[matched]
            if len(matched_dv_idx) == 0:
                return
            x_valid_matched = x_valid_matched[matched, :]
            is_active_valid_matched = is_active_valid_matched[matched, :]

        return matched_dv_idx[0]

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        """
        Return for each vector in x (n x nx) the valid discrete vector index.
        Design vectors may be valid, however canonical vectors are never asked to be corrected.
        """
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(correct_correct_x={self.correct_correct_x}, ' \
               f'random_if_multiple={self._random_if_multiple})'


class ClosestEagerCorrector(EagerCorrectorBase):
    """
    Eager corrector that corrects design vectors by matching them to the closest available canonical
    design vector, as measured by the Manhattan or Euclidean distance.
    Optionally distances are weighted to prefer changes on the right side of the design vectors.
    """

    def __init__(self, design_space: ArchDesignSpace, euclidean=False, correct_correct_x: bool = None,
                 random_if_multiple: bool = None):
        self.euclidean = euclidean
        super().__init__(design_space, correct_correct_x=correct_correct_x, random_if_multiple=random_if_multiple)

    def _get_corrected_x_idx(self, x: np.ndarray) -> np.ndarray:
        # Calculate distances from provided design vectors to canonical design vectors
        x_valid, is_active_valid = self.x_valid_active
        is_discrete_mask = self.is_discrete_mask
        x_valid_discrete = x_valid[:, is_discrete_mask]

        metric = 'euclidean' if self.euclidean else 'cityblock'
        weights = np.linspace(1.1, 1, x_valid_discrete.shape[1])
        x_dist = distance.cdist(x[:, is_discrete_mask], x_valid_discrete, metric=metric, w=weights)

        xi_canonical = np.zeros((x.shape[0],), dtype=int)
        for i, xi in enumerate(x):
            # Select vector with minimum distance
            min_dist_idx, = np.where(x_dist[i, :] == np.min(x_dist[i, :]))

            if len(min_dist_idx) > 1 and self._random_if_multiple:
                xi_canonical[i] = np.random.choice(min_dist_idx)
            else:
                xi_canonical[i] = min_dist_idx[0]
        return xi_canonical

    def __repr__(self):
        return f'{self.__class__.__name__}(correct_correct_x={self.correct_correct_x}, ' \
               f'random_if_multiple={self._random_if_multiple}, euclidean={self.euclidean})'
