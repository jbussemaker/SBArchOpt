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
import pandas as pd
from typing import List, Optional, Union, Tuple
from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.core.variable import Variable
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sb_arch_opt.design_space import ArchDesignSpace, ImplicitArchDesignSpace

__all__ = ['ArchOptProblemBase', 'ArchOptRepair', 'ArchDesignSpace']


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

    def __init__(self, des_vars: Union[List[Variable], ArchDesignSpace], n_obj=1, n_ieq_constr=0, n_eq_constr=0,
                 **kwargs):

        # Create a design space if we didn't get one
        if isinstance(des_vars, ArchDesignSpace):
            design_space = des_vars
        else:
            design_space = ImplicitArchDesignSpace(
                des_vars,
                self._correct_x,
                self._is_conditionally_active,
                self._get_n_valid_discrete,
                self._get_n_active_cont_mean,
                self._gen_all_discrete_x,
                self._get_n_correct_discrete,
                self._get_n_active_cont_mean_correct,
            )
        self.design_space = design_space

        n_var = design_space.n_var
        xl = design_space.xl
        xu = design_space.xu
        var_types = {f'DV{i}': des_var for i, des_var in enumerate(design_space.des_vars)}
        self.des_vars = design_space.des_vars

        super().__init__(n_var=n_var, xl=xl, xu=xu, vars=var_types,
                         n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, **kwargs)

    @property
    def is_cat_mask(self):
        """Boolean mask specifying for each design variable whether it is a categorical variable"""
        return self.design_space.is_cat_mask

    @property
    def is_int_mask(self):
        """Boolean mask specifying for each design variable whether it is an integer variable"""
        return self.design_space.is_int_mask

    @property
    def is_discrete_mask(self):
        """Boolean mask specifying for each design variable whether it is a discrete (i.e. integer or categorical)
        variable"""
        return self.design_space.is_discrete_mask

    @property
    def is_cont_mask(self):
        """Boolean mask specifying for each design variable whether it is a continues (i.e. not discrete) variable"""
        return self.design_space.is_cont_mask

    @property
    def is_conditionally_active(self):
        """Boolean mask specifying for each design variable whether it is conditionally active or not"""
        return self.design_space.is_conditionally_active

    def get_categorical_values(self, x: np.ndarray, i_dv) -> np.ndarray:
        """Gets the associated categorical variable values for some design variable"""
        return self.design_space.get_categorical_values(x, i_dv)

    def correct_x(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Imputes design vectors and returns activeness vectors"""
        return self.design_space.correct_x(x)

    def _correct_x_impute(self, x: np.ndarray, is_active: np.ndarray):
        self.design_space.correct_x_impute(x, is_active)

    def impute_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Applies the default imputation to design vectors:
        - Sets inactive discrete design variables to 0
        - Sets inactive continuous design variables to the mid of their bounds
        """
        self.design_space.impute_x(x, is_active)

    def get_n_valid_discrete(self) -> Optional[int]:
        """Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio"""
        return self.design_space.get_n_valid_discrete()

    def get_n_correct_discrete(self) -> Optional[int]:
        """Return the number of correct discrete design points (ignoring continuous dimensions); enables calculation of
        the correction ratio"""
        return self.design_space.get_n_correct_discrete()

    def get_n_declared_discrete(self) -> int:
        """Returns the number of declared discrete design points (ignoring continuous dimensions), calculated from the
        cartesian product of discrete design variables"""
        return self.design_space.get_n_declared_discrete()

    @property
    def all_discrete_x(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate all possible discrete design vectors, if the problem provides this function. Returns both the design
        vectors and activeness information. Active continuous variables may have any value within their bounds."""
        return self.design_space.all_discrete_x

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
        x_out: np.ndarray = x.copy()
        self.design_space.round_x_discrete(x_out)
        is_active_out = np.ones(x.shape, dtype=bool)

        f_out = np.zeros((x.shape[0], self.n_obj))*np.nan
        g_out = np.zeros((x.shape[0], self.n_ieq_constr))*np.nan
        h_out = np.zeros((x.shape[0], self.n_eq_constr))*np.nan

        # If the design space definition is explicit, it means that that is all we need to correct and impute, and we
        # prevent subsequent changing of the inputs
        if self.design_space.is_explicit():
            self._correct_x_impute(x_out, is_active_out)
            x_out.setflags(write=False)
            is_active_out.setflags(write=False)

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
        if len(pop_or_out) == 0:
            return np.array([], dtype=bool)

        f = pop_or_out.get('F')
        is_failed = np.any(~np.isfinite(f), axis=1)

        g = pop_or_out.get('G')
        if g is not None:
            is_failed |= np.any(~np.isfinite(g), axis=1)

        h = pop_or_out.get('H')
        if h is not None:
            is_failed |= np.any(~np.isfinite(h), axis=1)

        return is_failed

    @classmethod
    def get_population_statistics(cls, pop: Population, show=False):
        rows = []

        def _add_row(met_name, not_met_name, i_met):
            pop_met = pop_stat[i_met]
            n_met = len(pop_stat[i_met])

            i_not_met = np.delete(np.arange(len(pop_stat)), i_met)
            n_not_met = len(pop_stat[i_not_met])

            rows.append([met_name, n_met, f'{100*n_met/(len(pop) or 1):.1f}%',
                         not_met_name, n_not_met, f'{100*n_not_met/(len(pop) or 1):.1f}%'])
            return pop_met

        pop_stat = pop
        i_is_eval = np.array([i for i, ind in enumerate(pop_stat) if len(ind.evaluated or []) > 0], dtype=int)
        pop_stat = _add_row('Evaluated', 'Unknown', i_is_eval)
        pop_stat = _add_row('Viable', 'Failed', ~cls.get_failed_points(pop_stat))
        pop_stat = _add_row('Feasible', 'Infeasible',
                            pop_stat.get('feas') if len(pop_stat) > 0 else np.array([], dtype=bool))

        i_nds = np.array([], dtype=int)
        if 0 < len(pop_stat) < 2000:
            try:
                i_nds = NonDominatedSorting().do(pop_stat.get('F'), only_non_dominated_front=True)
            except IndexError:
                pass
        _add_row('Optimal', 'Dominated', i_nds)

        pop_stats = pd.DataFrame(data=rows, columns=pd.MultiIndex.from_tuples([
            ('condition', 'name'), ('condition', 'n'), ('condition', '%'),
            ('not met', 'name'), ('not met', 'n'), ('not met', '%'),
        ]))

        if show:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                   'display.expand_frame_repr', False, 'max_colwidth', None):
                print(pop_stats)

        return pop_stats

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
        discrete_imp_ratio = self.get_discrete_imputation_ratio()
        cont_imp_ratio = self.get_continuous_imputation_ratio()
        if not np.isnan(imp_ratio) or not np.isnan(discrete_imp_ratio):
            print(f'HIER         : {imp_ratio > 1 or discrete_imp_ratio > 1}')  # Is it a hierarchical problem?
            print(f'n_valid_discr: {self.get_n_valid_discrete()}')  # Number of valid discrete design points
            print(f'imp_ratio    : {imp_ratio:.2f} (discr.: {discrete_imp_ratio:.2f}; '
                  f'cont.: {cont_imp_ratio:.2f})')  # Imputation ratio: nr of declared designs / n_valid_discr

        corr_ratio = self.get_correction_ratio()
        discrete_corr_ratio = self.get_discrete_correction_ratio()
        cont_corr_ratio = self.get_continuous_correction_ratio()
        corr_fraction = self.design_space.correction_fraction
        if not np.isnan(corr_ratio) or not np.isnan(discrete_corr_ratio):
            # print(f'n_corr_discr : {self.get_n_correct_discrete()}')  # Number of correct discrete design points
            # Correction ratio: nr of declared designs / n_correct_discr
            print(f'corr_ratio   : {corr_ratio:.2f} (discr.: {discrete_corr_ratio:.2f}; '
                  f'cont.: {cont_corr_ratio:.2f}; fraction of imp_ratio: {corr_fraction*100:.1f}%)')

        fail_rate = self.get_failure_rate()
        if fail_rate is not None and fail_rate > 0:
            might_have_warn = ' (CHECK DECLARATION)' if not self.might_have_hidden_constraints() else ''
            # Problem has hidden constraints (= regions with failed evaluations)?
            print(f'HC           : {self.might_have_hidden_constraints()}{might_have_warn}')
            print(f'failure_rate : {fail_rate*100:.0f}%')  # Failure rate: fraction of points where evaluation fails

        self._print_extra_stats()

    def get_imputation_ratio(self) -> float:
        """
        Returns the problem-level imputation ratio, a measure of how hierarchical the problem is. It is calculated
        from the product of the discrete and continuous imputation ratios.
        """
        return self.design_space.imputation_ratio

    def get_discrete_imputation_ratio(self) -> float:
        """
        Returns the imputation ratio considering only the discrete design vectors: it represents the ratio between
        number of declared discrete dimensions (Cartesian product) and the number of valid discrete design vectors.
        A value of 1 indicates no hierarchy, any value higher than 1 means there is hierarchy and the higher the value,
        the more difficult it is to randomly sample a valid design vector.
        """
        return self.design_space.discrete_imputation_ratio

    def get_continuous_imputation_ratio(self) -> float:
        """
        Returns the imputation ratio considering only the continuous design variables: it represents the nr of
        continuous dimensions over the mean number of active continuous dimensions, as seen over all valid discrete
        design vectors. The higher the number, the less continuous dimensions are active on average. A value of 1
        indicates all continuous dimensions are always active.
        """
        return self.design_space.continuous_imputation_ratio

    def get_correction_ratio(self) -> float:
        """
        Returns the problem-level correction ratio, a measure of how much of the imputation ratio is due to a need for
        correction (i.e. value constraints).
        It is calculated from the product of the discrete and continuous correction ratios.
        """
        return self.design_space.correction_ratio

    def get_discrete_correction_ratio(self) -> float:
        """
        Returns the correction ratio considering only the discrete design vectors: it represents the ratio between
        number of declared discrete dimensions (Cartesian product) and the number of correct discrete design vectors.
        A value of 1 indicates no correction is needed, any value higher than 1 means correction is needed and the
        higher the value, the more difficult it is to randomly sample a correct design vector.
        """
        return self.design_space.discrete_correction_ratio

    def get_continuous_correction_ratio(self) -> float:
        """
        Returns the correction ratio considering only the continuous design variables: it represents the nr of
        continuous dimensions over the mean number of active continuous dimensions, as seen over all correct discrete
        design vectors. The higher the number, the less continuous dimensions are active on average. A value of 1
        indicates all continuous dimensions are always active.
        """
        return self.design_space.continuous_correction_ratio

    def get_discrete_rates(self, force=False, show=False) -> Optional[pd.DataFrame]:
        """Returns for each discrete value of the discrete design variables, how often the relatively occur over all
        possible design vectors. A value of -1 represents an inactive design variable. Results are returned in a
        pandas DataFrame with each column representing a design variable.
        Also adds a measure of rate diversity: difference between lowest and highest occurring values."""
        return self.design_space.get_discrete_rates(force=force, show=show)

    """##############################
    ### IMPLEMENT FUNCTIONS BELOW ###
    ##############################"""

    def _is_conditionally_active(self) -> List[bool]:
        """Return for each design variable whether it is conditionally active (i.e. might become inactive). Not needed
        if an explicit design space is provided."""

    def _get_n_valid_discrete(self) -> int:
        """
        Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio.
        Valid discrete design points are discrete design points where value constraints are satisfied and where
        inactive design variables are imputed/canonical (compare _get_n_correct_discrete).
        """

    def _get_n_active_cont_mean(self) -> float:
        """
        Get the mean number of active continuous dimensions, as seen over all valid discrete design vectors.

        For example, if there are two valid discrete design vectors like this:
        x_discrete x_continuous1 x_continuous2
        0          Active        Active
        1          Active        Inactive

        Then the mean number of active continuous dimensions is:
        3 (total nr of active continuous dimensions) / 2 (number of discrete vectors) = 1.5
        """

    def _get_n_correct_discrete(self) -> Optional[int]:
        """
        Return the number of correct discrete design points (ignoring continuous dimensions); enables calculation of
        the correction ratio.
        Correct discrete design points are discrete design points where value constraints are satisfied, where however
        inactive design variables can have any value (compare _get_n_valid_discrete).
        """

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

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate all possible discrete design vectors (if available). Returns design vectors and activeness
        information. Not needed if an explicit design space is provided."""

    def store_results(self, results_folder):
        """Callback function to store intermediate or final results in some results folder. Should include all
        previously evaluated design points."""

    def load_previous_results(self, results_folder) -> Optional[Population]:
        """Return a Population (with X and F (optionally G and H) defined) created from previous results."""

    def might_have_hidden_constraints(self):
        """By default, it is assumed that at any time one or more points might fail to evaluate (i.e. return NaN).
        If you are sure this will never happen, set this to False. This information can be used by optimization
        algorithms to speed up the process."""
        return True

    def get_failure_rate(self) -> float:
        """Estimate the failure rate: the fraction of randomly-sampled points of which evaluation will fail"""

    def _print_extra_stats(self):
        """Print extra statistics when the print_stats() function is used"""

    def get_n_batch_evaluate(self) -> Optional[int]:
        """If the problem evaluation benefits from parallel batch process, return the appropriate batch size here"""

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        """
        Implement evaluation and write results in the provided output matrices:
        - x (design vectors): discrete variables have integer values, imputed design vectors can be output here (except
                              if using an explicit design space definition)
        - is_active (activeness): vector specifying for each design variable whether it was active or not
        - f (objectives): written as a minimization
        - g (inequality constraints): written as "<= 0"
        - h (equality constraints): written as "= 0"
        """
        raise NotImplementedError

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix and (if needed) correct any design variables that are partially inactive.
        Imputation of inactive design variables is always applied after this function.

        Only needed if no explicit design space model is given.
        Only used if not all discrete design vectors `all_discrete_x` is available OR
        `self.design_space.use_auto_corrector = False` OR `self.design_space.needs_cont_correction = True`:
        --> set `self.design_space.use_auto_corrector = False` to prevent using an automatic corrector
        --> set `self.design_space.needs_cont_correction = True` if automatic correction can be used but also continuous
            variables might have to be corrected (the automatic corrector only corrects discrete variables)
        """

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
