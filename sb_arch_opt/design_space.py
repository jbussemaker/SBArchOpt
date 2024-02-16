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
import itertools
import numpy as np
import pandas as pd
from typing import *
from cached_property import cached_property
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.variable import Variable, Real, Integer, Binary, Choice

__all__ = ['ArchDesignSpace', 'ImplicitArchDesignSpace', 'CorrectorInterface', 'CorrectorUnavailableError']


class CorrectorInterface:
    """
    Interface for an object implementing some problem-agnostic correction behavior.
    """

    def correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        No need to impute inactive design variables.
        """
        raise NotImplementedError


class CorrectorUnavailableError(RuntimeError):
    pass


class ArchDesignSpace:
    """
    Base class for a hierarchical architecture design space definition. The basic information optimization algorithms
    need from a design space is as follows:

    - Design variable definition: types, bounds, options
    - Some way to exhaustively sample all discrete design vectors (aka full factorial; grid)
    - Activeness information: for a given matrix of design vectors, a boolean matrix specifying which vars are active
    - Imputation: correction of design vectors to canonical design vectors, setting inactive variables to some default
      value and correcting invalid variable values
    - Optionally calculate the design of the design space: number of valid discrete design vectors

    Design variable terminology:

    - Continuous: any value between some lower and upper bound (inclusive)
      --> for example [0, 1]: 0, 0.25, .667, 1
    - Discrete: integer or categorical
    - Integer: integer between 0 and some upper bound (inclusive); ordering and distance matters
      --> for example [0, 2]: 0, 1, 2
    - Categorical: one of n options, encoded as integers; ordering and distance are not defined
      --> for example [red, blue, yellow]: red, blue
    """

    def __init__(self):
        self._choice_value_map = None
        self._is_initialized = False
        self.use_auto_corrector = False
        self.needs_cont_correction = False

    @cached_property
    def n_var(self):
        return len(self.des_vars)

    @cached_property
    def des_vars(self) -> List[Variable]:
        """
        Returns the defined design variables.
        Categorical variables (Choice) are encoded as integer values from 0 to n_opts-1. Use get_categorical_values to
        get the associated categorical values.
        """
        corr_des_vars = []
        self._choice_value_map = choice_value_map = {}
        for i_var, var_type in enumerate(self._get_variables()):

            if isinstance(var_type, Choice):
                choice_value_map[i_var] = var_type.options
                var_type = Choice(options=list(range(len(var_type.options))))

            elif not isinstance(var_type, (Real, Integer, Binary)):
                raise RuntimeError(f'Unsupported variable type: {var_type!r}')

            corr_des_vars.append(var_type)

        self._is_initialized = True
        return corr_des_vars

    @cached_property
    def is_conditionally_active(self) -> np.ndarray:
        """
        Returns a mask specifying for each design variable whether it is conditionally active (i.e. may become inactive
        at some point).
        """

        is_cond_active = self._is_conditionally_active()

        # If not provided, deduce from all discrete design vectors
        if is_cond_active is None:
            _, is_act_all = self.all_discrete_x
            if is_act_all is not None:
                return np.any(~is_act_all, axis=0)

            raise RuntimeError('Could not deduce is_conditionally_active from all x, '
                               'implement _is_conditionally_active!')

        is_cond_active = np.array(is_cond_active)
        if len(is_cond_active) != self.n_var:
            raise ValueError(f'is_cont_active should be same length as x: {len(is_cond_active)} != {self.n_var}')
        if np.all(is_cond_active):
            raise ValueError(f'At least one variable should be nonconditionally active: {is_cond_active}')

        return is_cond_active

    def get_categorical_values(self, x: np.ndarray, i_dv) -> np.ndarray:
        """Gets the associated categorical variable values for some design variable"""
        if not self._is_initialized:
            getattr(self, 'des_vars')
        if i_dv not in self._choice_value_map:
            raise ValueError(f'Design variable is not categorical: {i_dv}')

        x_values = x[:, i_dv]
        values = x_values.astype(str)
        for i_cat, value in enumerate(self._choice_value_map[i_dv]):
            values[x_values == i_cat] = value
        return values

    @cached_property
    def xl(self) -> np.ndarray:
        """Vector containing lower bounds of variables"""
        xl = np.zeros((self.n_var,))
        for i_var, des_var in enumerate(self.des_vars):
            if isinstance(des_var, (Real, Integer)):
                xl[i_var] = des_var.bounds[0]
        return xl

    @cached_property
    def x_mid(self) -> np.ndarray:
        """Mid-bounds values"""
        return .5*(self.xl+self.xu)

    @cached_property
    def xu(self) -> np.ndarray:
        """Vector containing upper bounds of variables"""
        xu = np.empty((self.n_var,))
        for i_var, des_var in enumerate(self.des_vars):
            if isinstance(des_var, (Real, Integer)):
                xu[i_var] = des_var.bounds[1]
            elif isinstance(des_var, Binary):
                xu[i_var] = 1
            elif isinstance(des_var, Choice):
                xu[i_var] = len(des_var.options)-1
        return xu

    @cached_property
    def is_int_mask(self) -> np.ndarray:
        """Boolean vector specifying whether each variable is an integer (ordinal) variable"""
        return np.array([isinstance(des_var, (Integer, Binary)) for des_var in self.des_vars], dtype=bool)

    @cached_property
    def is_cat_mask(self) -> np.ndarray:
        """Boolean vector specifying whether each variable is a categorical variable"""
        return np.array([isinstance(des_var, Choice) for des_var in self.des_vars], dtype=bool)

    @cached_property
    def is_discrete_mask(self) -> np.ndarray:
        """Boolean vector specifying whether each variable is a discrete (integer or categorical) variable"""
        return self.is_int_mask | self.is_cat_mask

    @cached_property
    def is_cont_mask(self) -> np.ndarray:
        """Boolean vector specifying whether each variable is a continuous variable"""
        return ~self.is_discrete_mask

    def correct_x(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Imputes design vectors and returns activeness vectors"""
        x_imputed = x.copy()
        self.round_x_discrete(x_imputed)
        is_active = np.ones(x.shape, dtype=bool)

        self.correct_x_impute(x_imputed, is_active)
        return x_imputed, is_active

    def correct_x_impute(self, x: np.ndarray, is_active: np.ndarray):
        """Corrects and imputes design vectors, assuming that they have already been corrected for discreteness"""
        self._correct_x_corrector(x, is_active)
        self.impute_x(x, is_active)

    def _correct_x_corrector(self, x: np.ndarray, is_active: np.ndarray):
        """Corrects design vectors and fills is_active matrix by a corrector if available,
        otherwise uses the problem-specific correction mechanism"""

        corrector = self.corrector
        if corrector is not None and self.use_auto_corrector:
            try:
                corrector.correct_x(x, is_active)

                # The corrector only corrects discrete variables, check if correction of continuous variables is needed
                if not self.needs_cont_correction:
                    return

            except CorrectorUnavailableError:
                pass

        self._correct_x(x, is_active)

    @cached_property
    def corrector(self) -> Optional[CorrectorInterface]:
        """
        Correction algorithm for problem-agnostic optimal correction in case all design vectors (`all_discrete_x`) are
        available. Set `use_auto_corrector = False` to force to use the problem-specific `_correct_x` function.
        """
        return self._get_corrector()

    def _get_corrector(self) -> Optional[CorrectorInterface]:
        """Get the default corrector algorithm"""
        from sb_arch_opt.correction import ClosestEagerCorrector
        return ClosestEagerCorrector(self)

    def round_x_discrete(self, x: np.ndarray):
        """
        Ensures that discrete design variables take up integer values.
        Rounding is not applied directly, as this would reduce the amount of points assigned to the first and last
        options.

        Directly rounding:
        np.unique(np.round(np.linspace(0, 2, 100)).astype(int), return_counts=True) --> 25, 50, 25 (center bias)

        Stretched rounding:
        x = np.linspace(0, 2, 100)
        xs = x*((np.max(x)+.99)/np.max(x))-.5
        np.unique(np.abs(np.round(xs)).astype(int), return_counts=True) --> 34, 33, 33 (evenly distributed)
        """
        is_discrete_mask = self.is_discrete_mask
        x_discrete = x[:, is_discrete_mask].astype(float)
        xl, xu = self.xl[is_discrete_mask], self.xu[is_discrete_mask]
        diff = xu-xl

        for ix in range(x_discrete.shape[1]):
            x_discrete[x_discrete[:, ix] < xl[ix], ix] = xl[ix]
            x_discrete[x_discrete[:, ix] > xu[ix], ix] = xu[ix]

        x_stretched = (x_discrete-xl)*((diff+.9999)/diff)-.5
        x_rounded = (np.round(x_stretched)+xl).astype(int)

        x[:, is_discrete_mask] = x_rounded

    def impute_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Applies the default imputation to design vectors:
        - Sets inactive discrete design variables to 0
        - Sets inactive continuous design variables to the mid of their bounds
        """

        # Impute inactive discrete design variables: set to their lower bound
        for i_dv in np.where(self.is_discrete_mask)[0]:
            x[~is_active[:, i_dv], i_dv] = self.xl[i_dv]

        # Impute inactive continuous design variables: set to their mid-bound
        x_mid = self.x_mid
        for i_dv in np.where(self.is_cont_mask)[0]:
            x[~is_active[:, i_dv], i_dv] = x_mid[i_dv]

    def get_n_valid_discrete(self) -> Optional[int]:
        """Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio"""
        n_valid = self._get_n_valid_discrete()
        if n_valid is not None:
            return n_valid

        x_discrete, _ = self.all_discrete_x
        if x_discrete is not None:
            return x_discrete.shape[0]

    def get_n_correct_discrete(self) -> Optional[int]:
        """Return the number of correct design points (ignoring continuous dimensions); enables calculation of the
        correction ratio"""
        n_correct = self._get_n_correct_discrete()
        if n_correct is not None:
            return n_correct

        all_discrete_x_n_correct = self.all_discrete_x_n_correct
        if all_discrete_x_n_correct is not None:
            return int(np.sum(all_discrete_x_n_correct))

    def get_n_declared_discrete(self) -> int:
        """Returns the number of declared discrete design points (ignoring continuous dimensions), calculated from the
        cartesian product of discrete design variables"""

        # Get number of discrete options for each discrete design variable
        is_discrete_mask = self.is_discrete_mask
        n_opts_discrete = self.xu[is_discrete_mask]-self.xl[is_discrete_mask]+1
        if len(n_opts_discrete) == 0:
            return 1

        return int(np.prod(n_opts_discrete, dtype=float))

    @cached_property
    def imputation_ratio(self) -> float:
        """
        Returns the problem-level imputation ratio, a measure of how hierarchical the problem is. It is calculated
        from the product of the discrete and continuous imputation ratios.
        """
        return self.discrete_imputation_ratio * self.continuous_imputation_ratio

    @cached_property
    def discrete_imputation_ratio(self) -> float:
        """
        Returns the imputation ratio considering only the discrete design vectors: it represents the ratio between
        number of declared discrete dimensions (Cartesian product) and the number of valid discrete design vectors.
        A value of 1 indicates no hierarchy, any value higher than 1 means there is hierarchy and the higher the value,
        the more difficult it is to randomly sample a valid design vector.
        """

        # Get valid design points
        n_valid = self.get_n_valid_discrete()
        if n_valid is None:
            return np.nan
        if n_valid == 0:
            return 1.

        n_declared = self.get_n_declared_discrete()
        discrete_imp_ratio = n_declared/n_valid
        return discrete_imp_ratio

    @cached_property
    def continuous_imputation_ratio(self) -> float:
        """
        Returns the imputation ratio considering only the continuous design variables: it represents the nr of
        continuous dimensions over the mean number of active continuous dimensions, as seen over all valid discrete
        design vectors. The higher the number, the less continuous dimensions are active on average. A value of 1
        indicates all continuous dimensions are always active.
        """

        # Check if we have any continuous dimensions
        i_is_cont = np.where(self.is_cont_mask)[0]
        if len(i_is_cont) == 0:
            return 1.

        # Check if mean active continuous dimensions is explicitly defined
        n_cont_active_mean = self._get_n_active_cont_mean()

        # Get from discrete design vectors
        if n_cont_active_mean is None:
            x_all, is_active_all = self.all_discrete_x
            if x_all is None:
                return np.nan

            n_cont_active_mean = np.sum(is_active_all[:, i_is_cont]) / x_all.shape[0]

        # Calculate imputation ratio from declared / mean_active
        n_cont_dim_declared = len(i_is_cont)
        return n_cont_dim_declared / n_cont_active_mean

    @cached_property
    def correction_ratio(self) -> float:
        """
        Returns the problem-level correction ratio, a measure of how much of the imputation ratio is due to a need for
        correction (i.e. value constraints).
        It is calculated from the product of the discrete and continuous correction ratios.
        """
        return self.discrete_correction_ratio * self.continuous_correction_ratio

    @cached_property
    def discrete_correction_ratio(self) -> float:
        """
        Returns the correction ratio considering only the discrete design vectors: it represents the ratio between
        number of declared discrete dimensions (Cartesian product) and the number of correct discrete design vectors.
        A value of 1 indicates no correction is needed, any value higher than 1 means correction is needed and the
        higher the value, the more difficult it is to randomly sample a correct design vector.
        """

        # Get correct design points
        n_correct = self.get_n_correct_discrete()
        if n_correct is None:
            return np.nan
        if n_correct == 0:
            return 1.

        n_declared = self.get_n_declared_discrete()
        discrete_imp_ratio = n_declared/n_correct
        return discrete_imp_ratio

    @cached_property
    def continuous_correction_ratio(self) -> float:
        """
        Returns the correction ratio considering only the continuous design variables: it represents the nr of
        continuous dimensions over the mean number of active continuous dimensions, as seen over all correct discrete
        design vectors. The higher the number, the less continuous dimensions are active on average. A value of 1
        indicates all continuous dimensions are always active.
        """

        # Check if we have any continuous dimensions
        i_is_cont = np.where(self.is_cont_mask)[0]
        if len(i_is_cont) == 0:
            return 1.

        # Check if mean active continuous dimensions is explicitly defined
        n_cont_active_mean = self._get_n_active_cont_mean_correct()

        # Get from discrete design vectors
        if n_cont_active_mean is None:
            x_all, is_active_all = self.all_discrete_x
            n_correct = self.all_discrete_x_n_correct
            if x_all is None or n_correct is None:
                return np.nan

            is_active_weighted = is_active_all[:, i_is_cont].astype(float) * np.array([n_correct]).T
            n_cont_active_mean = np.sum(is_active_weighted) / np.sum(n_correct)

        # Calculate imputation ratio from declared / mean_active
        n_cont_dim_declared = len(i_is_cont)
        return n_cont_dim_declared / n_cont_active_mean

    @cached_property
    def correction_fraction(self) -> float:
        """The fraction of design space hierarchy (quantified by the imputation ratio) that is due to the need for
        correction (i.e. value constraints)."""
        imputation_ratio = self.imputation_ratio
        if imputation_ratio == 1.:
            return 0.
        return np.log10(self.correction_ratio) / np.log10(imputation_ratio)

    def get_discrete_rates(self, force=False, show=False) -> Optional[pd.DataFrame]:
        """Returns for each discrete value of the discrete design variables, how often the relatively occur over all
        possible design vectors. A value of -1 represents an inactive design variable. Results are returned in a
        pandas DataFrame with each column representing a design variable.
        Also adds a measure of rate diversity: difference between lowest and highest occurring values."""

        # Get all discrete design vectors
        x_all, is_act_all = self.all_discrete_x
        if x_all is None:
            if not force:
                return
            x_all, is_act_all = self.all_discrete_x_by_trial_and_imputation

        df = self.calculate_discrete_rates(x_all-self.xl, is_act_all)

        if show:
            is_discrete_mask = np.concatenate([self.is_discrete_mask, [True]])
            with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                   'display.expand_frame_repr', False, 'max_colwidth', None):
                print(df.iloc[:, is_discrete_mask].replace(np.nan, ''))
        return df

    @staticmethod
    def calculate_discrete_rates_raw(x: np.ndarray, is_active: np.ndarray, is_discrete_mask: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, set]:
        # Ignore All-Nan slice warning
        import warnings
        warnings.filterwarnings('ignore', 'All-NaN.*', RuntimeWarning)

        # x should be moved to 0!
        x_merged = x.astype(int)+1
        x_merged[~is_active] = 0
        n = x_merged.shape[0]

        i_opts = set()
        if np.all(~is_discrete_mask):
            counts = np.zeros((1, x.shape[1]))*np.nan
        else:
            # Count the values
            counts = np.zeros((int(np.max(x[:, is_discrete_mask]))+2, x.shape[1]))*np.nan
            for ix in range(x.shape[1]):
                if not is_discrete_mask[ix]:
                    continue

                counts_i = np.bincount(x_merged[:, ix])
                for iv, count in enumerate(counts_i):
                    if count > 0:
                        counts[int(iv), ix] = count/n
                        i_opts.add(iv-1)

        # Calculate diversity metric: the range between the lowest and highest occurring values
        diversity = np.nanmax(counts, axis=0) - np.nanmin(counts, axis=0)
        if -1 in i_opts:
            active_counts = counts[1:, :]
            active_counts /= np.nansum(active_counts, axis=0)
            active_diversity = np.nanmax(active_counts, axis=0) - np.nanmin(active_counts, axis=0)
        else:
            active_diversity = diversity

        return counts, diversity, active_diversity, i_opts

    def calculate_discrete_rates(self, x: np.ndarray, is_active: np.ndarray) -> pd.DataFrame:

        is_discrete_mask = self.is_discrete_mask
        counts, diversity, active_diversity, i_opts = self.calculate_discrete_rates_raw(x, is_active, is_discrete_mask)

        # Create dataframe
        has_value = np.array([iv-1 in i_opts for iv in range(counts.shape[0])])
        counts = counts[has_value, :]

        columns = [f'x{ix}' for ix in range(x.shape[1])]
        df = pd.DataFrame(index=sorted(list(i_opts)), columns=columns, data=counts)
        df = df.rename(index={val: 'inactive' if val == -1 else f'opt {val}' for val in df.index})

        is_cat_mask = self.is_cat_mask
        x_type = [('cat' if is_cat_mask[ix] else 'int') if is_discrete_mask[ix] else 'cont'
                  for ix in range(x.shape[1])]

        df = pd.concat([
            df,
            pd.Series(index=columns, data=diversity, name='diversity').to_frame().T,
            pd.Series(index=columns, data=active_diversity, name='active-diversity').to_frame().T,
            pd.Series(index=columns, data=x_type, name='x_type').to_frame().T,
            pd.Series(index=columns, data=self.is_conditionally_active, name='is_cond').to_frame().T,
        ], axis=0)

        max_diversity = np.zeros((len(df),))*np.nan
        max_diversity[-4] = df.iloc[-4, :].max()
        max_diversity[-3] = df.iloc[-3, :].max()
        df = pd.concat([df, pd.Series(index=df.index, data=max_diversity, name='max')], axis=1)
        return df

    def quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n design vectors (also return is_active) without generating all design vectors first"""

        x, is_active = self._quick_sample_discrete_x(n)
        if x.shape[1] != self.n_var or is_active.shape[1] != self.n_var:
            raise RuntimeError(f'Inconsistent design vector dimensions: {x.shape[1]} != {self.n_var}')
        if x.shape[0] > n:
            x = x[:n, :]
            is_active = is_active[:n, :]
        x = x.astype(float)  # Otherwise continuous variables cannot be imputed

        self.round_x_discrete(x)
        self.impute_x(x, is_active)

        return x, is_active

    @cached_property
    def all_discrete_x(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate all possible discrete design vectors, if the problem provides this function. Returns both the design
        vectors and activeness information. Active continuous variables may have any value within their bounds."""

        # Check if this problem implements discrete design vector generation
        discrete_x = self._gen_all_discrete_x()
        if discrete_x is None:
            return None, None

        # Impute values (mostly for continuous dimensions)
        x, is_active = discrete_x
        if x is None or is_active is None:
            return None, None
        if x.shape[1] != self.n_var or is_active.shape[1] != self.n_var:
            raise RuntimeError(f'Inconsistent design vector dimensions: {x.shape[1]} != {self.n_var}')
        x = x.astype(float)  # Otherwise continuous variables cannot be imputed
        self.impute_x(x, is_active)

        # Cross-check with numerical estimate
        n_valid = self._get_n_valid_discrete()
        if n_valid is not None and (n_valid != x.shape[0] or n_valid != is_active.shape[0]):
            raise RuntimeError(f'Inconsistent estimation of nr of discrete design vectors: {n_valid} != {x.shape[0]}')

        return x, is_active

    @cached_property
    def all_discrete_x_n_correct(self) -> Optional[np.ndarray]:
        """For each valid discrete design vector, get the number of correct versions (i.e. design vectors where inactive
        variables have any value)"""
        _, is_active = self.all_discrete_x
        if is_active is None:
            return

        # Get nr of discrete options for each discrete design variable
        is_discrete_mask = self.is_discrete_mask
        is_inactive_discrete = ~is_active[:, is_discrete_mask]
        if is_inactive_discrete.shape[0] == 0 or is_inactive_discrete.shape[1] == 0:
            return np.ones((is_inactive_discrete.shape[0],), dtype=int)
        n_opts_discrete = self.xu[is_discrete_mask]-self.xl[is_discrete_mask]+1

        # For each valid discrete vector, get the number of correct versions
        n_correct = np.ones(is_inactive_discrete.shape)
        for j in range(is_inactive_discrete.shape[1]):
            n_correct[is_inactive_discrete[:, j], j] = n_opts_discrete[j]

        n_correct = np.prod(n_correct, axis=1).astype(int)
        return n_correct

    @cached_property
    def all_discrete_x_by_trial_and_imputation(self):
        """
        Find all possible discrete design vectors by trail and imputation:
        - Generate the Cartesian product of all discrete variables
        - Impute design vectors
        - Remove duplicates
        """
        return self._get_all_discrete_x_by_trial_and_imputation()

    def _get_all_discrete_x_by_trial_and_imputation(self):
        # First sample only discrete dimensions
        opt_values = self.get_exhaustive_sample_values(n_cont=1)
        x_cart_product_gen = itertools.product(*opt_values)

        is_cont_mask = self.is_cont_mask
        is_discrete_mask = ~is_cont_mask

        use_auto_corrector = self.use_auto_corrector
        self.use_auto_corrector = False

        try:
            # Create and repair the sampled design vectors in batches
            n_batch = 1000
            x_discr = np.zeros((0, len(opt_values)))
            is_act_discr = np.zeros(x_discr.shape, dtype=bool)
            while True:
                # Get next batch
                x_repair = []
                for _ in range(n_batch):
                    x_next = next(x_cart_product_gen, None)
                    if x_next is None:
                        break
                    x_repair.append(x_next)
                if len(x_repair) == 0:
                    break
                x_repair = np.array(x_repair)

                # Repair current batch
                # print(f'Sampling {x_repair.shape[0]} ({x_repaired.shape[0]} sampled)')
                x_repair_input = x_repair
                x_repair, is_active = self.correct_x(x_repair)

                # Remove repaired points
                is_not_repaired = ~np.any(x_repair[:, is_discrete_mask] != x_repair_input[:, is_discrete_mask], axis=1)
                x_repair = x_repair[is_not_repaired, :]
                is_active = is_active[is_not_repaired, :]

                x_discr = np.row_stack([x_discr, x_repair])
                is_act_discr = np.row_stack([is_act_discr, is_active.astype(bool)])

            # Impute continuous values
            self.impute_x(x_discr, is_act_discr)

        finally:
            self.use_auto_corrector = use_auto_corrector

        # Use these results for subsequent calls of all_discrete_x
        all_discrete_x = self.__dict__.get('all_discrete_x')
        if all_discrete_x is None or all_discrete_x[0] is None:
            self.__dict__['all_discrete_x'] = (x_discr, is_act_discr)

        return x_discr, is_act_discr

    def get_exhaustive_sample_values(self, n_cont=5):
        # Determine bounds and which design variables are discrete
        xl, xu = self.xl, self.xu
        is_cont = self.is_cont_mask

        # Get values to be sampled for each design variable
        return [np.linspace(xl[i], xu[i], n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1) for i in range(len(xl))]

    def _quick_random_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback sampling if hierarchical sampling is not available"""
        from sb_arch_opt.problem import ArchOptProblemBase

        stub_problem = ArchOptProblemBase(self)
        x = LatinHypercubeSampling().do(stub_problem, n).get('X')
        self.round_x_discrete(x)

        is_active = np.ones(x.shape, dtype=bool)
        self._correct_x(x, is_active)
        return x, is_active

    def is_explicit(self) -> bool:
        """Whether this design space is defined explicitly, that is: a model of the design space is available and
        correct, and therefore the problem evaluation function never needs to correct any design vector"""
        raise NotImplementedError

    def _get_variables(self) -> List[Variable]:
        """Returns the list of design variables (pymoo classes)"""
        raise NotImplementedError

    def _is_conditionally_active(self) -> Optional[List[bool]]:
        """Returns for each design variable whether it is conditionally active (i.e. may become inactive)"""
        raise NotImplementedError

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        Imputation of inactive variables is handled automatically.
        """
        raise NotImplementedError

    def _quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n discrete design vectors (also return is_active) without generating all design vectors first"""
        raise NotImplementedError

    def _get_n_valid_discrete(self) -> Optional[int]:
        """
        Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio.
        Valid discrete design points are discrete design points where value constraints are satisfied and where
        inactive design variables are imputed/canonical (compare _get_n_correct_discrete).
        """
        raise NotImplementedError

    def _get_n_active_cont_mean(self) -> Optional[float]:
        """
        Get the mean number of active continuous dimensions, as seen over all valid discrete design vectors.

        For example, if there are two valid discrete design vectors like this:
        x_discrete x_continuous1 x_continuous2
        0          Active        Active
        1          Active        Inactive

        Then the mean number of active continuous dimensions is:
        3 (total nr of active continuous dimensions) / 2 (number of discrete vectors) = 1.5
        """
        raise NotImplementedError

    def _get_n_correct_discrete(self) -> Optional[int]:
        """
        Return the number of correct discrete design points (ignoring continuous dimensions); enables calculation of
        the correction ratio.
        Correct discrete design points are discrete design points where value constraints are satisfied, where however
        inactive design variables can have any value (compare _get_n_valid_discrete).
        """
        raise NotImplementedError

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        """
        Get the mean number of active continuous dimensions, as seen over all valid discrete design vectors.

        For example, if there are two correct discrete design vectors like this:
        x_discrete x_continuous1 x_continuous2
        0          Active        Active
        1          Active        Inactive
        2          Active        Inactive

        Then the mean number of active continuous dimensions is:
        4 (total nr of active continuous dimensions) / 3 (number of discrete vectors) = 1.3333...
        """
        raise NotImplementedError

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate all possible discrete design vectors (if available). Returns design vectors and activeness
        information."""
        raise NotImplementedError


class ImplicitArchDesignSpace(ArchDesignSpace):
    """An implicit, problem-specific definition of the architecture design space"""

    def __init__(self, des_vars: List[Variable], correct_x_func: Callable[[np.ndarray, np.ndarray], None],
                 is_conditional_func: Callable[[], List[bool]],
                 n_valid_discrete_func: Callable[[], int] = None, n_active_cont_mean: Callable[[], float] = None,
                 gen_all_discrete_x_func: Callable[[], Optional[Tuple[np.ndarray, np.ndarray]]] = None,
                 n_correct_discrete_func: Callable[[], int] = None,
                 n_active_cont_mean_correct: Callable[[], float] = None):
        self._variables = des_vars
        self._correct_x_func = correct_x_func
        self._is_conditional_func = is_conditional_func
        self._n_valid_discrete_func = n_valid_discrete_func
        self._n_active_cont_mean = n_active_cont_mean
        self._n_correct_discrete_func = n_correct_discrete_func
        self._n_active_cont_mean_correct = n_active_cont_mean_correct
        self._gen_all_discrete_x_func = gen_all_discrete_x_func
        super().__init__()

    def is_explicit(self) -> bool:
        return False

    def _get_variables(self) -> List[Variable]:
        return self._variables

    def _is_conditionally_active(self) -> Optional[List[bool]]:
        return self._is_conditional_func()

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        self._correct_x_func(x, is_active)

    def _quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._quick_random_sample_discrete_x(n)

    def _get_n_valid_discrete(self) -> Optional[int]:
        if self._n_valid_discrete_func is not None:
            return self._n_valid_discrete_func()

    def _get_n_active_cont_mean(self) -> Optional[float]:
        if self._n_active_cont_mean is not None:
            return self._n_active_cont_mean()

    def _get_n_correct_discrete(self) -> Optional[int]:
        if self._n_correct_discrete_func is not None:
            return self._n_correct_discrete_func()

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        if self._n_active_cont_mean_correct is not None:
            return self._n_active_cont_mean_correct()

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._gen_all_discrete_x_func is not None:
            return self._gen_all_discrete_x_func()
