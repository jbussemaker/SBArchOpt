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
import numpy as np
from typing import List, Optional, Union
from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.core.variable import Variable, Real, Integer, Choice, Binary

__all__ = ['ArchOptProblemBase', 'ArchOptRepair']


class ArchOptProblemBase(Problem):
    """
    Base class for an architecture optimization problem, featuring:
    - Mixed-discrete design variable definitions
    - Function for imputing a design vector and requesting the activeness vector (specifying which variables were active
      for the imputed design vector)
    - Interface to get a repair operator to implement design vector imputation

    Design variable terminology:
    - Continuous: any value between some lower and upper bound (inclusive)
      --> for example [0, 1]: 0, 0.25, .667, 1
    - Discrete: integer or categorical
    - Integer: integer between 0 and some upper bound (inclusive); ordering and distance matters
      --> for example [0, 2]: 0, 1, 2
    - Categorical: one of n options, encoded as integers; ordering and distance are not defined
      --> for example [red, blue, yellow]: red, blue

    Note that objectives (F) are always defined as minimization, inequality constraints (G) are satisfied when equal or
    lower than 0, and equality constraints (H) are satisfied when equal to 0.
    Conversion and normalization should be implemented in the evaluation function.

    Also note that we are not exactly following the mixed-variable approach
    [suggested by pymoo](https://pymoo.org/customization/mixed.html), but rather keeping the design vectors in a matrix
    with all different variable types in there. To facilitate this, categorical variables are encoded as integers, and
    the mixed-variable operators have been rewritten in SBArchOpt.
    """

    def __init__(self, des_vars: List[Variable], n_obj=1, n_ieq_constr=0, n_eq_constr=0, **kwargs):

        # Harmonize the pymoo variable definition interface
        n_var = len(des_vars)
        xl = np.zeros((n_var,))
        xu = np.empty((n_var,))
        self._is_int_mask = is_int_mask = np.zeros((n_var,), dtype=bool)
        self._is_cat_mask = is_cat_mask = np.zeros((n_var,), dtype=bool)
        self._choice_value_map = {}
        corr_des_vars = []
        for i_var, var_type in enumerate(des_vars):
            if isinstance(var_type, Real):
                xl[i_var], xu[i_var] = var_type.bounds

            elif isinstance(var_type, Integer):
                is_int_mask[i_var] = True
                xl[i_var], xu[i_var] = var_type.bounds

            elif isinstance(var_type, Binary):
                is_int_mask[i_var] = True
                xu[i_var] = 1

            elif isinstance(var_type, Choice):
                is_cat_mask[i_var] = True
                xu[i_var] = len(var_type.options)-1
                self._choice_value_map[i_var] = var_type.options
                var_type = Choice(options=list(range(len(var_type.options))))

            else:
                raise RuntimeError(f'Unsupported variable type: {var_type!r}')
            corr_des_vars.append(var_type)

        self.des_vars = corr_des_vars
        var_types = {f'DV{i}': des_var for i, des_var in enumerate(corr_des_vars)}

        self._is_discrete_mask = is_int_mask | is_cat_mask
        self._is_cont_mask = ~self._is_discrete_mask
        self._x_cont_mid = .5*(xl[self._is_cont_mask]+xu[self._is_cont_mask])

        super().__init__(n_var=n_var, xl=xl, xu=xu, vars=var_types,
                         n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, **kwargs)

    @property
    def is_cat_mask(self):
        """Boolean mask specifying for each design variable whether it is a categorical variable"""
        return self._is_cat_mask

    @property
    def is_int_mask(self):
        """Boolean mask specifying for each design variable whether it is an integer variable"""
        return self._is_int_mask

    @property
    def is_discrete_mask(self):
        """Boolean mask specifying for each design variable whether it is a discrete (i.e. integer or categorical)
        variable"""
        return self._is_discrete_mask

    @property
    def is_cont_mask(self):
        """Boolean mask specifying for each design variable whether it is a continues (i.e. not discrete) variable"""
        return self._is_cont_mask

    def get_categorical_values(self, x: np.ndarray, i_dv) -> np.ndarray:
        """Gets the associated categorical variable values for some design variable"""
        if i_dv not in self._choice_value_map:
            raise ValueError(f'Design variable is not categorical: {i_dv}')

        x_values = x[:, i_dv]
        values = x_values.astype(np.str)
        for i_cat, value in enumerate(self._choice_value_map[i_dv]):
            values[x_values == i_cat] = value
        return values

    def correct_x(self, x: np.ndarray):
        """Imputes design vectors and returns activeness vectors"""
        x_imputed = x.copy()
        self._correct_x_discrete(x_imputed)
        is_active = np.ones(x.shape, dtype=bool)

        self._correct_x_impute(x_imputed, is_active)
        return x_imputed, is_active

    def _correct_x_impute(self, x: np.ndarray, is_active: np.ndarray):
        self._correct_x(x, is_active)
        self._impute_x(x, is_active)

    def _correct_x_discrete(self, x: np.ndarray):
        """Ensures that discrete design variables take up integer values"""
        x[:, self._is_discrete_mask] = np.round(x[:, self._is_discrete_mask].astype(np.float64)).astype(np.int)

    def _impute_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Applies the default imputation to design vectors:
        - Sets inactive discrete design variables to 0
        - Sets inactive continuous design variables to the mid of their bounds
        """

        # Impute inactive discrete design variables
        for i_dv in np.where(self.is_discrete_mask)[0]:
            x[~is_active[:, i_dv], i_dv] = self.xl[i_dv]

        # Impute inactive continuous design variables
        for i_cont, i_dv in enumerate(np.where(self.is_cont_mask)[0]):
            x[~is_active[:, i_dv], i_dv] = self._x_cont_mid[i_cont]

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates a set of design vectors (provided as matrix). Outputs:
        X: imputed design vectors
        is_active: activeness matrix, specifying for each design variable whether it is active or not
        F: objective values
        G: inequality constraint values
        H: equality constraint values
        """
        # Prepare output matrices for evaluation
        x_out = x.copy()
        self._correct_x_discrete(x_out)
        is_active_out = np.ones(x.shape, dtype=bool)

        f_out = np.zeros((x.shape[0], self.n_obj))*np.nan
        g_out = np.zeros((x.shape[0], self.n_ieq_constr))*np.nan
        h_out = np.zeros((x.shape[0], self.n_eq_constr))*np.nan

        # Call evaluation function
        self._arch_evaluate(x_out, is_active_out, f_out, g_out, h_out, *args, **kwargs)

        # Provide outputs to pymoo
        out['X'] = x_out
        out['is_active'] = is_active_out
        out['F'] = f_out
        if self.n_ieq_constr > 0:
            out['G'] = g_out
        if self.n_eq_constr > 0:
            out['H'] = h_out

    @staticmethod
    def get_failed_points(pop_or_out: Union[dict, Population]):
        f = pop_or_out.get('F')
        is_failed = np.any(~np.isfinite(f), axis=1)

        g = pop_or_out.get('G')
        if g is not None:
            is_failed |= np.any(~np.isfinite(g), axis=1)

        h = pop_or_out.get('H')
        if h is not None:
            is_failed |= np.any(~np.isfinite(h), axis=1)

        return is_failed

    def extend_pop_data(self, population: Population) -> Population:
        """Extend the data in the Population (assuming at least X and F are present): impute X and provide is_active"""
        population = population.copy()
        x_imp, is_active = self.correct_x(population.get('X'))
        population.set('X', x_imp)
        population.set('is_active', is_active)
        return population

    @staticmethod
    def get_repair():
        """Get the repair operator for architecture optimization problems"""
        return ArchOptRepair()

    def print_stats(self):
        n_discr = np.sum(self.is_discrete_mask)
        n_cont = np.sum(self.is_cont_mask)
        try:
            print(f'problem: {self!r}')
        except NotImplementedError:
            pass
        print(f'n_discr: {n_discr}')  # Number of discrete design variables
        print(f'n_cont : {n_cont}')  # Number of continuous design variables
        print(f'n_obj  : {self.n_obj}')  # Number of objectives
        print(f'n_con  : {self.n_ieq_constr}')  # Number of (inequality) constraints
        print(f'MD     : {n_discr > 0 and n_cont > 0}')  # Is it a mixed-discrete problem?
        print(f'MO     : {self.n_obj > 1}')  # Is it a multi-objective problem?

        imp_ratio = self.get_imputation_ratio()
        if not np.isnan(imp_ratio):
            print(f'HIER         : {imp_ratio > 1}')  # Is it a hierarchical problem?
            print(f'n_valid_discr: {self.get_n_valid_discrete()}')  # Number of valid discrete design points
            print(f'imp_ratio    : {imp_ratio:.2f}')  # Imputation ratio: nr of declared designs / n_valid_discr

        fail_rate = self.get_failure_rate()
        if fail_rate is not None and fail_rate > 0:
            might_have_warn = ' (CHECK DECLARATION)' if not self.might_have_hidden_constraints() else ''
            # Problem has hidden constraints (= regions with failed evaluations)?
            print(f'HC           : {self.might_have_hidden_constraints()}{might_have_warn}')
            print(f'failure_rate : {fail_rate*100:.0f}%')  # Failure rate: fraction of points where evaluation fails

        self._print_extra_stats()

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

    def get_n_declared_discrete(self) -> int:
        """Returns the number of declared discrete design points (ignoring continuous dimensions), calculated from the
        cartesian product of discrete design variables"""

        # Get number of discrete options for each discrete design variable
        n_opts_discrete = self.xu[self._is_discrete_mask]-self.xl[self._is_discrete_mask]+1
        if len(n_opts_discrete) == 0:
            return 1

        return int(np.prod(n_opts_discrete, dtype=np.float))

    """##############################
    ### IMPLEMENT FUNCTIONS BELOW ###
    ##############################"""

    def get_n_valid_discrete(self) -> int:
        """Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio"""

    def store_results(self, results_folder, final=False):
        """Callback function to store intermediate or final results in some results folder"""

    def load_previous_results(self, results_folder) -> Optional[Population]:
        """
        Return a Population (with X and F (optionally G and H) defined) created from previous results.
        Call `extend_pop_data` afterwards if is_active and is_failed are also needed as data.
        """

    def might_have_hidden_constraints(self):
        """By default, it is assumed that at any time one or more points might fail to evaluate (i.e. return NaN).
        If you are sure this will never happen, set this to False. This information can be used by optimization
        algorithms to speed up the process."""
        return True

    def get_failure_rate(self) -> float:
        """Estimate the failure rate: the fraction of randomly-sampled points of which evaluation will fail"""

    def _print_extra_stats(self):
        """Print extra statistics when the print_stats() function is used"""

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """
        Implement evaluation and write results in the provided output matrices:
        - x (design vectors): discrete variables have integer values, imputed design vectors can be output here
        - is_active (activeness): vector specifying for each design variable whether it was active or not
        - f (objectives): written as a minimization
        - g (inequality constraints): written as "<= 0"
        - h (equality constraints): written as "= 0"
        """
        raise NotImplementedError

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """Fill the activeness matrix and (if needed) impute any design variables that are partially inactive.
        Imputation of inactive design variables is always applied after this function."""
        raise NotImplementedError

    def __repr__(self):
        """repr() of the class, should be unique for unique Pareto fronts"""
        raise NotImplementedError


class ArchOptRepair(Repair):
    """
    Repair operator for architecture optimization problems.
    Stores the repaired activeness vector under `latest_is_active`.
    """

    def __init__(self):
        self.latest_is_active = None
        super().__init__()

    def do(self, problem, pop, **kwargs):
        is_array = not isinstance(pop, Population)
        x = pop if is_array else pop.get('X')

        if isinstance(problem, ArchOptProblemBase):
            x, self.latest_is_active = problem.correct_x(x)
        else:
            self.latest_is_active = None

        if is_array:
            return x
        pop.set('X', x)
        return pop

    def __repr__(self):
        return f'{self.__class__.__name__}()'
