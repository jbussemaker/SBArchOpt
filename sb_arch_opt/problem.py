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
from typing import List
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
    """

    def __init__(self, var_types: List[Variable], n_obj=1, n_ieq_constr=0, n_eq_constr=0, **kwargs):

        # Harmonize the pymoo variable definition interface
        n_var = len(var_types)
        xl = np.zeros((n_var,))
        xu = np.empty((n_var,))
        self._is_int_mask = is_int_mask = np.zeros((n_var,), dtype=bool)
        self._is_cat_mask = is_cat_mask = np.zeros((n_var,), dtype=bool)
        self._choice_value_map = {}
        corr_var_types = []
        for i_var, var_type in enumerate(var_types):
            if isinstance(var_type, Real):
                xl[i_var], xu[i_var] = var_type.bounds

            elif isinstance(var_type, Integer):
                is_int_mask[i_var] = True
                xu[i_var] = var_type.bounds[1]

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
            corr_var_types.append(var_type)

        self._is_discrete_mask = is_int_mask | is_cat_mask
        self._is_cont_mask = ~self._is_discrete_mask

        super().__init__(n_var=n_var, xl=xl, xu=xu, vars=corr_var_types,
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

    def get_categorical_values(self, i_dv, x_i: np.ndarray) -> np.ndarray:
        """Gets the associated categorical variable values for some design variable"""
        if i_dv not in self._choice_value_map:
            raise ValueError(f'Design variable is not categorical: {i_dv}')

        values = x_i.astype(np.str)
        for i_cat, value in enumerate(self._choice_value_map[i_dv]):
            values[x_i == i_cat] = value
        return values

    def correct_x(self, x: np.ndarray):
        """Imputes design vectors and returns activeness vectors"""
        x_imputed = x.copy()
        self._correct_x_discrete(x_imputed)
        is_active = np.ones(x.shape, dtype=bool)

        self._correct_x(x_imputed, is_active)
        return x_imputed, is_active

    def _correct_x_discrete(self, x: np.ndarray):
        """Ensures that discrete design variables take up integer values"""
        x[:, self._is_discrete_mask] = np.round(x[:, self._is_discrete_mask].astype(np.float64)).astype(np.int)

    def _evaluate(self, x, out, *args, **kwargs):
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
    def get_repair():
        """Get the repair operator for architecture optimization problems"""
        return ArchOptRepair()

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
        """Impute the design vectors (discrete variables already have integer values),
        writing the imputed design vectors and the activeness matrix to the provided matrices"""
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
