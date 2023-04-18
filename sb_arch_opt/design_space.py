"""
Licensed under the GNU General Public License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/gpl-3.0.html.en

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""
import itertools
import numpy as np
import pandas as pd
from typing import *
from cached_property import cached_property
from pymoo.core.variable import Variable, Real, Integer, Binary, Choice

__all__ = ['ArchDesignSpace', 'ImplicitArchDesignSpace']


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

    def get_categorical_values(self, x: np.ndarray, i_dv) -> np.ndarray:
        """Gets the associated categorical variable values for some design variable"""
        if not self._is_initialized:
            getattr(self, 'des_vars')
        if i_dv not in self._choice_value_map:
            raise ValueError(f'Design variable is not categorical: {i_dv}')

        x_values = x[:, i_dv]
        values = x_values.astype(np.str)
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
        self._correct_x(x, is_active)
        self.impute_x(x, is_active)

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
        x_discrete = x[:, is_discrete_mask].astype(np.float64)
        xl, xu = self.xl[is_discrete_mask], self.xu[is_discrete_mask]
        diff = xu-xl

        for ix in range(x_discrete.shape[1]):
            x_discrete[x_discrete[:, ix] < xl[ix], ix] = xl[ix]
            x_discrete[x_discrete[:, ix] > xu[ix], ix] = xu[ix]

        x_stretched = (x_discrete-xl)*((diff+.99)/diff)-.5
        x_rounded = (np.round(x_stretched)+xl).astype(np.int)

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

    def get_n_declared_discrete(self) -> int:
        """Returns the number of declared discrete design points (ignoring continuous dimensions), calculated from the
        cartesian product of discrete design variables"""

        # Get number of discrete options for each discrete design variable
        is_discrete_mask = self.is_discrete_mask
        n_opts_discrete = self.xu[is_discrete_mask]-self.xl[is_discrete_mask]+1
        if len(n_opts_discrete) == 0:
            return 1

        return int(np.prod(n_opts_discrete, dtype=np.float))

    def get_imputation_ratio(self) -> float:
        """
        Returns the ratio between declared and valid design points; gives an estimate on how much design variable
        hierarchy plays a role for this problem. A value of 1 means there is no hierarchy, any value higher than 1
        means there is hierarchy. The higher the value, the more difficult it is to "randomly" sample a valid design
        vector (e.g. imputation ratio = 10 means that 1/10th of declared design vectors is valid).
        """

        # Get valid design points
        n_valid = self.get_n_valid_discrete()
        if n_valid is None:
            return np.nan
        if n_valid == 0:
            return 1.

        n_declared = self.get_n_declared_discrete()
        imp_ratio = n_declared/n_valid
        return imp_ratio

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

        # Set inactive values to -1
        x_merged = (x_all-self.xl).astype(int)
        x_merged[~is_act_all] = -1
        n = x_merged.shape[0]

        # Count the values
        is_discrete_mask = self.is_discrete_mask
        counts = {}
        i_opts = set()
        for ix in range(len(is_discrete_mask)):
            if not is_discrete_mask[ix]:
                counts[f'x{ix}'] = {}
                continue

            values, counts_i = np.unique(x_merged[:, ix], return_counts=True)
            i_opts |= set(values)
            counts[f'x{ix}'] = {value: counts_i[iv]/n for iv, value in enumerate(values)}

        df = pd.DataFrame(index=sorted(list(i_opts)), columns=list(counts.keys()), data=counts)
        df = df.rename(index={val: 'inactive' if val == -1 else f'opt {val}' for val in df.index})

        # Add a measure of diversity: the range between the lowest and highest occurring values
        diversity = df.max(axis=0)-df.min(axis=0)
        if -1 in i_opts:
            df_active = df.iloc[1:, :]
            col_sums = df_active.sum(axis=0)
            df_active /= col_sums
            active_diversity = df_active.max(axis=0)-df_active.min(axis=0)
        else:
            active_diversity = diversity

        df = pd.concat([df, pd.Series(diversity, name='diversity').to_frame().T,
                        pd.Series(active_diversity, name='active-diversity').to_frame().T], axis=0)

        if show:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                   'display.expand_frame_repr', False, 'max_colwidth', -1):
                print(df.iloc[:, self.is_discrete_mask].replace(np.nan, ''))

        return df

    def quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n design vectors (also return is_active) without generating all design vectors first"""

        x, is_active = self._quick_sample_discrete_x(n)
        if x.shape[1] != self.n_var or is_active.shape[1] != self.n_var:
            raise RuntimeError(f'Inconsistent design vector dimensions: {x.shape[1]} != {self.n_var}')
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
    def all_discrete_x_by_trial_and_imputation(self):
        """
        Find all possible discrete design vectors by trail and imputation:
        - Generate the Cartesian product of all discrete variables
        - Impute design vectors
        - Remove duplicates
        """

        # First sample only discrete dimensions
        opt_values = self.get_exhaustive_sample_values(n_cont=1)
        x_cart_product_gen = itertools.product(*opt_values)

        is_cont_mask = self.is_cont_mask
        is_discrete_mask = ~is_cont_mask

        # Create and repair the sampled design vectors in batches
        n_batch = 1000
        x_repaired = np.zeros((0, len(opt_values)), dtype=int)
        is_active_repaired = np.zeros(x_repaired.shape, dtype=bool)
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
            x_repair = np.array(x_repair).astype(int)

            # Repair current batch
            # print(f'Sampling {x_repair.shape[0]} ({x_repaired.shape[0]} sampled)')
            x_repair_input = x_repair
            x_repair, is_active = self.correct_x(x_repair)

            # Remove repaired points
            is_not_repaired = ~np.any(x_repair[:, is_discrete_mask] != x_repair_input[:, is_discrete_mask], axis=1)
            x_repair = x_repair[is_not_repaired, :]
            is_active = is_active[is_not_repaired, :]

            x_repaired = np.row_stack([x_repaired, x_repair])
            is_active_repaired = np.row_stack([is_active_repaired, is_active.astype(bool)])

        x_discr = np.row_stack(x_repaired).astype(float)
        is_act_discr = np.row_stack(is_active_repaired)

        # Impute continuous values
        self.impute_x(x_discr, is_act_discr)

        return x_discr, is_act_discr

    def get_exhaustive_sample_values(self, n_cont=5):
        # Determine bounds and which design variables are discrete
        xl, xu = self.xl, self.xu
        is_cont = self.is_cont_mask

        # Get values to be sampled for each design variable
        return [np.linspace(xl[i], xu[i], n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1) for i in range(len(xl))]

    def is_explicit(self) -> bool:
        """Whether this design space is defined explicitly, that is: a model of the design space is available and
        correct, and therefore the problem evaluation function never needs to correct any design vector"""
        raise NotImplementedError

    def _get_variables(self) -> List[Variable]:
        """Returns the list of design variables (pymoo classes)"""
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
        """Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio"""
        raise NotImplementedError

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate all possible discrete design vectors (if available). Returns design vectors and activeness
        information."""
        raise NotImplementedError


class ImplicitArchDesignSpace(ArchDesignSpace):
    """An implicit, problem-specific definition of the architecture design space"""

    def __init__(self, des_vars: List[Variable], correct_x_func: Callable[[np.ndarray, np.ndarray], None],
                 n_valid_discrete_func: Callable[[], int] = None,
                 gen_all_discrete_x_func: Callable[[], Optional[Tuple[np.ndarray, np.ndarray]]] = None):
        self._variables = des_vars
        self._correct_x_func = correct_x_func
        self._n_valid_discrete_func = n_valid_discrete_func
        self._gen_all_discrete_x_func = gen_all_discrete_x_func
        super().__init__()

    def is_explicit(self) -> bool:
        return False

    def _get_variables(self) -> List[Variable]:
        return self._variables

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        self._correct_x_func(x, is_active)

    def _quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        opt_values = self.get_exhaustive_sample_values(n_cont=1)
        x = np.empty((n, self.n_var))
        is_discrete_mask = self.is_discrete_mask
        for i_dv in range(self.n_var):
            if is_discrete_mask[i_dv]:
                i_opt_sampled = np.random.choice(len(opt_values[i_dv]), n, replace=True)
                x[:, i_dv] = opt_values[i_dv][i_opt_sampled]

        is_active = np.ones(x.shape, dtype=bool)
        self._correct_x(x, is_active)
        return x, is_active

    def _get_n_valid_discrete(self) -> Optional[int]:
        if self._n_valid_discrete_func is not None:
            return self._n_valid_discrete_func()

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._gen_all_discrete_x_func is not None:
            return self._gen_all_discrete_x_func()
