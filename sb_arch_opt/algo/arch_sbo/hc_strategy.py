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
import timeit
import logging
import numpy as np
from typing import *
from sb_arch_opt.problem import *
from sb_arch_opt.algo.arch_sbo.models import *
from pymoo.util.normalization import Normalization, SimpleZeroToOneNormalization

try:
    from smt.surrogate_models.surrogate_model import SurrogateModel
    assert HAS_ARCH_SBO
except ImportError:
    assert not HAS_ARCH_SBO

try:
    from sklearn.ensemble import RandomForestClassifier as RFC
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

__all__ = ['get_hc_strategy', 'HiddenConstraintStrategy', 'PredictionHCStrategy', 'PredictorInterface',
           'SKLearnClassifier', 'RandomForestClassifier', 'SMTPredictor', 'MDGPRegressor', 'HAS_SKLEARN',
           'RejectionHCStrategy', 'ReplacementHCStrategyBase', 'GlobalWorstReplacement']

log = logging.getLogger('sb_arch_opt.sbo_hc')


def get_hc_strategy(kpls_n_dim: Optional[int] = 10, min_pov: float = .25):
    """
    Get a hidden constraints strategy that works well for most problems.

    The minimum Probability of Viability (min_pov) can be used to determine how more the algorithm will be pushed
    towards exploration over exploitation. Values between 10% and 50% are shown to give good optimization results.
    """

    # Get the predictor: RF works best but requires scikit-learn
    try:
        predictor = RandomForestClassifier(n=100, n_dim=10)
    except ImportError:
        predictor = MDGPRegressor(kpls_n_dim=kpls_n_dim)

    # Create the strategy: use as additional constraint at Probability of Validity >= 50%
    return PredictionHCStrategy(predictor, constraint=True, min_pov=min_pov)


class HiddenConstraintStrategy:
    """
    Base class for implementing a strategy for dealing with hidden constraints.
    """

    @staticmethod
    def is_failed(y: np.ndarray):
        return np.any(~np.isfinite(y), axis=1)

    def initialize(self, problem: ArchOptProblemBase):
        pass

    def mod_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Modify inputs and outputs for the surrogate model used for the main infill function"""
        return x, y

    def prepare_infill_search(self, x: np.ndarray, y: np.ndarray):
        """Prepare infill search given the (non-modified) normalized inputs and outputs"""

    def adds_infill_constraint(self) -> bool:
        """Whether the strategy adds an inequality constraint to the infill search problem"""
        return False

    def evaluate_infill_constraint(self, x: np.ndarray) -> np.ndarray:
        """If the problem added an infill constraint, evaluate it here, returning an nx-length vector"""

    def mod_infill_objectives(self, x: np.ndarray, f_infill: np.ndarray) -> np.ndarray:
        """Modify the infill objectives (in-place)"""
        return f_infill

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class RejectionHCStrategy(HiddenConstraintStrategy):
    """Strategy that simply rejects failed points before training the model"""

    def mod_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Remove failed points from the training set
        is_not_failed = ~self.is_failed(y)
        return x[is_not_failed, :], y[is_not_failed, :]

    def __str__(self):
        return 'Rejection'

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ReplacementHCStrategyBase(HiddenConstraintStrategy):
    """Base class for a strategy that replaces failed outputs by some value"""

    def __init__(self):
        self._normalization: Optional[SimpleZeroToOneNormalization] = None
        super().__init__()

    def initialize(self, problem: ArchOptProblemBase):
        self._normalization = SimpleZeroToOneNormalization(xl=problem.xl, xu=problem.xu)

    def mod_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Separate into failed and valid (non-failed) set
        is_failed = self.is_failed(y)
        x_valid = x[~is_failed, :]
        y_valid = y[~is_failed, :]
        x_failed = x[is_failed, :]
        y_failed = y[is_failed, :]

        # If there are no failed points, no need to replace
        if x_failed.shape[0] == 0:
            return x, y

        # If there are no valid points, replace with 1
        if y_valid.shape[0] == 0:
            y_failed_replace = np.ones(y_failed.shape)
        else:
            y_failed_replace = self._replace_y(x_failed, y_failed, x_valid, y_valid)

        # Replace values
        y = y.copy()
        y[is_failed, :] = y_failed_replace
        return x, y

    def _replace_y(self, x_failed: np.ndarray, y_failed: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) \
            -> np.ndarray:
        """Return values for replacing y_failed (x values are normalized)"""
        raise NotImplementedError

    def get_replacement_strategy_name(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return f'Replacement: {self.get_replacement_strategy_name()}'

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class GlobalWorstReplacement(ReplacementHCStrategyBase):
    """Replace failed values with the worst values known for these outputs"""

    def _replace_y(self, x_failed: np.ndarray, y_failed: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) \
            -> np.ndarray:
        # Get global worst values
        y_worst = np.max(y_valid, axis=0)

        # Replace
        y_replace = np.zeros(y_failed.shape)+y_worst
        return y_replace

    def get_replacement_strategy_name(self) -> str:
        return 'Global Worst'


class PredictorInterface:
    """Interface class for some validity predictor"""
    _training_doe = {}
    _reset_pickle_keys = []

    def __init__(self):
        self.training_set = None
        self._normalization: Optional[Normalization] = None
        self._trained_single_class = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._reset_pickle_keys:
            if key in state:
                state[key] = None
        return state

    def initialize(self, problem: ArchOptProblemBase):
        self._normalization = self._get_normalization(problem)
        self._initialize(problem)

    def _get_normalization(self, problem: ArchOptProblemBase) -> Normalization:
        return SimpleZeroToOneNormalization(xl=problem.xl, xu=problem.xu, estimate_bounds=False)

    def _initialize(self, problem: ArchOptProblemBase):
        pass

    def train(self, x: np.ndarray, y_is_valid: np.ndarray):
        # Check if we are training a classifier with only 1 class
        self._trained_single_class = single_class = y_is_valid[0] if len(set(y_is_valid)) == 1 else None

        if single_class is None:
            log.debug(f'Training hidden constraints predictor {self!s}; x size: {x.shape}')
            s = timeit.default_timer()

            self._train(x, y_is_valid)

            train_time = timeit.default_timer()-s
            log.debug(f'Trained hidden constraints predictor in {train_time:.2f} seconds')

    def evaluate_probability_of_validity(self, x: np.ndarray) -> np.ndarray:
        if self._trained_single_class is not None:
            return np.ones((x.shape[0],))*self._trained_single_class

        return self._evaluate_probability_of_validity(x)

    def _train(self, x: np.ndarray, y_is_valid: np.ndarray):
        """Train the model (x's are not normalized), y_is_valid is a vector"""
        raise NotImplementedError

    def _evaluate_probability_of_validity(self, x: np.ndarray) -> np.ndarray:
        """Get the probability of validity (0 to 1) at nx points (x is not normalized); should return a vector!"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class PredictionHCStrategy(HiddenConstraintStrategy):
    """Base class for a strategy that predictions where failed regions occur"""

    def __init__(self, predictor: PredictorInterface, constraint=True, min_pov=.5):
        check_dependencies()
        self.predictor = predictor
        self.constraint = constraint
        self.min_pov = min_pov
        super().__init__()

    def initialize(self, problem: ArchOptProblemBase):
        self.predictor.initialize(problem)

    def mod_xy_train(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Remove failed points form the training set
        is_not_failed = ~self.is_failed(y)
        return x[is_not_failed, :], y[is_not_failed, :]

    def prepare_infill_search(self, x: np.ndarray, y: np.ndarray):
        is_failed = self.is_failed(y)
        y_is_valid = (~is_failed).astype(float)
        self.predictor.train(x, y_is_valid)
        self.predictor.training_set = (x, y_is_valid)

    def adds_infill_constraint(self) -> bool:
        return self.constraint

    def evaluate_infill_constraint(self, x: np.ndarray) -> np.ndarray:
        pov = self.predictor.evaluate_probability_of_validity(x)
        pov = np.clip(pov, 0, 1)
        return self.min_pov-pov

    def mod_infill_objectives(self, x: np.ndarray, f_infill: np.ndarray) -> np.ndarray:
        pov = self.predictor.evaluate_probability_of_validity(x)
        pov = np.clip(pov, 0, 1)

        # The infill objectives are a minimization of some value between 0 and 1:
        # - The function-based infills (prediction mean), the underlying surrogates are trained on normalized y values
        # - The expected improvement is normalized between 0 and 1, where 1 corresponds to no expected improvement
        return 1-((1-f_infill).T*pov).T

    def __str__(self):
        type_str = 'G' if self.constraint else 'F'
        type_str += f' min_pov={self.min_pov}' if self.constraint and self.min_pov != .5 else ''
        return f'Prediction {type_str}: {self.predictor!s}'

    def __repr__(self):
        min_pov_str = f', min_pov={self.min_pov}' if self.constraint else ''
        return f'{self.__class__.__name__}({self.predictor!r}, constraint={self.constraint}{min_pov_str})'


class SKLearnClassifier(PredictorInterface):
    _reset_pickle_keys = ['_predictor']

    def __init__(self):
        self._predictor = None
        super().__init__()

    def _evaluate_probability_of_validity(self, x: np.ndarray) -> np.ndarray:
        if self._predictor is None:
            return np.ones((x.shape[0],))

        x_norm = self._normalization.forward(x)
        pov = self._predictor.predict_proba(x_norm)[:, 1]  # Probability of belonging to class 1 (valid points)
        return pov[:, 0] if len(pov.shape) == 2 else pov

    def _train(self, x: np.ndarray, y_is_valid: np.ndarray):
        self._do_train(self._normalization.forward(x), y_is_valid)

    def _do_train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class RandomForestClassifier(SKLearnClassifier):

    def __init__(self, n: int = 100, n_dim: float = None):
        self.n = n
        self.n_dim = n_dim
        super().__init__()

    def _do_train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        from sklearn.ensemble import RandomForestClassifier

        n_estimators = self.n
        if self.n_dim is not None:
            n_estimators = max(int(self.n_dim*x_norm.shape[1]), n_estimators)

        self._predictor = clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(x_norm, y_is_valid)

    def __str__(self):
        n_dim_str = f' | x{self.n_dim}' if self.n_dim is not None else ''
        return f'RFC ({self.n}{n_dim_str})'

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self.n})'


class SMTPredictor(PredictorInterface):
    _reset_pickle_keys = ['_model']

    def __init__(self):
        self._model: Optional['SurrogateModel'] = None
        super().__init__()

    def _evaluate_probability_of_validity(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict_values(self._normalization.forward(x))[:, 0]

    def _train(self, x: np.ndarray, y_is_valid: np.ndarray):
        self._do_train(self._normalization.forward(x), np.array([y_is_valid]).T)

    def _do_train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MDGPRegressor(SMTPredictor):
    """Uses SMT's mixed-discrete Kriging regressor"""

    def __init__(self, kpls_n_dim: Optional[int] = 10):
        self._problem = None
        self._kpls_n_dim = kpls_n_dim
        super().__init__()

    def _get_normalization(self, problem: ArchOptProblemBase) -> Normalization:
        return ModelFactory(problem).get_md_normalization()

    def _initialize(self, problem: ArchOptProblemBase):
        self._problem = problem

    def _do_train(self, x_norm: np.ndarray, y_is_valid: np.ndarray):
        kwargs = {}
        if self._kpls_n_dim is not None and x_norm.shape[1] > self._kpls_n_dim:
            kwargs['kpls_n_comp'] = self._kpls_n_dim

        model, _ = ModelFactory(self._problem).get_md_kriging_model(
            corr='abs_exp', theta0=[1e-2], n_start=5, **kwargs)
        self._model = model
        model.set_training_values(x_norm, y_is_valid)
        model.train()

    def __str__(self):
        return 'MD-GP'
