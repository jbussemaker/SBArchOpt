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
from typing import *
from sb_arch_opt.design_space import ArchDesignSpace
from sb_arch_opt.util import get_np_random_singleton
from pymoo.core.variable import Variable, Real, Integer, Choice

from ConfigSpace.util import generate_grid, get_random_neighbor
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition, NotEqualsCondition, GreaterThanCondition, LessThanCondition,\
    InCondition, AndConjunction, OrConjunction, ConditionComponent
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenInClause, ForbiddenEqualsRelation,\
    ForbiddenGreaterThanRelation, ForbiddenLessThanRelation, ForbiddenAndConjunction, AbstractForbiddenComponent

__all__ = [
    'ArchDesignSpace', 'ExplicitArchDesignSpace', 'ContinuousParam', 'IntegerParam', 'CategoricalParam', 'ParamType',
    'EqualsCondition', 'NotEqualsCondition', 'GreaterThanCondition', 'LessThanCondition', 'InCondition',
    'AndConjunction', 'OrConjunction',
    'ForbiddenEqualsClause', 'ForbiddenInClause', 'ForbiddenEqualsRelation', 'ForbiddenGreaterThanRelation',
    'ForbiddenLessThanRelation', 'ForbiddenAndConjunction',
]


class ContinuousParam(UniformFloatHyperparameter):
    """Specifies a continuous (float) design variable"""

    def __init__(self, name: str, lower: float, upper: float):
        super().__init__(name, lower=lower, upper=upper)


class IntegerParam(UniformIntegerHyperparameter):
    """Specifies an integer design variable"""

    def __init__(self, name: str, lower: int, upper: int):
        super().__init__(name, lower=lower, upper=upper)

    def get_neighbors(self, value: float, rs: np.random.RandomState, number: int = 4,
                      transform: bool = False, std: float = .2) -> List[int]:
        # Temporary fix until https://github.com/automl/ConfigSpace/pull/313 is merged
        center = self._transform(value)
        lower, upper = self.lower, self.upper
        n_neighbors = upper - lower - 1

        neighbors = []
        if n_neighbors < number:
            for v in range(lower, center):
                neighbors.append(v)
            for v in range(center+1, upper+1):  # Bugfix
                neighbors.append(v)

            if transform:
                return neighbors
            return self._inverse_transform(np.asarray(neighbors)).tolist()

        return super().get_neighbors(value, rs, number=number, transform=transform, std=std)


class CategoricalParam(CategoricalHyperparameter):
    """Specifies a categorical design variable"""

    def __init__(self, name: str, options: List[Union[str, float, int]]):
        super().__init__(name, choices=options)


ParamType = Union[ContinuousParam, IntegerParam, CategoricalParam]


class ExplicitArchDesignSpace(ArchDesignSpace):
    """
    A design space defined explicitly, by specifying conditionals and forbidden parameter combinations. Variables can
    either be float, integer, or categorical. Conditional statements activate variables based on some condition applied
    to some other variable, for example variable B is active when variable A has value x. Forbidden parameter
    combinations are used to specify when certain options become unavailable due to some condition.

    Usage:
    - Initialize the class with a list of parameters or add parameters after initialization
      --> use ContinuousParam, IntegerParam, CategoricalParam to specify parameters

    - Add conditions between parameters to conditionally activate design variables:

      ds.add_condition(EqualsCondition(ds['b'], ds['a'], 1))  # Activate b if a == 1

      --> refer to https://automl.github.io/ConfigSpace/main/api/conditions.html for more details

    - Add forbidden clauses to explicitly forbid the occurence of certain conditions (option values)

      ds.add_forbidden_clause(ForbiddenAndConjunction(
        ForbiddenEqualsClause(ds['a'], 1),
        ForbiddenEqualsClause(ds['b'], 2),
      ))  # Forbid a == 1 and b == 2 from occurring simultaneously

      --> refer to https://automl.github.io/ConfigSpace/main/api/forbidden_clauses.html for more details

    Under the hood, this explicit definition uses [ConfigSpace](https://automl.github.io/ConfigSpace/), a Python library
    for modeling hierarchical or conditional design spaces for hyperparameter optimization.

    Low-level access to the ConfigurationSpace object is possible, however should only be used for querying model
    structure. Hyperparameters are named according to their index, e.g. x0, x1, x2, etc.
    Original sorting order is maintained.
    """

    def __init__(self, params: List[ParamType] = None):
        super().__init__()

        self._var_names = []
        self._cs_idx = np.array([])
        self._inv_cs_idx = np.array([])

        self._cs = NoDefaultConfigurationSpace(name='Explicit DS')
        if params is not None:
            self.add_params(params)

    @property
    def config_space(self):
        return self._cs

    @property
    def cs_idx(self) -> np.ndarray:
        """Maps design space var index to underlying config space index"""
        return self._cs_idx

    @property
    def inv_cs_idx(self) -> np.ndarray:
        return self._inv_cs_idx

    def _update_cs_idx(self):
        cs_param_names = self._cs.get_hyperparameter_names()
        self._cs_idx = cs_idx = np.array([cs_param_names.index(name) for name in self._var_names])
        self._inv_cs_idx = inv_cs_idx = np.empty((len(cs_idx),), dtype=int)
        inv_cs_idx[cs_idx] = np.arange(len(cs_idx))

    def __iter__(self):
        return iter(self._cs)

    def __len__(self):
        return len(self._cs)

    def get(self, item: str, default=None):
        return self._cs.get(item, default=default)

    def __contains__(self, item):
        return item in self._cs

    def __getitem__(self, item) -> ParamType:
        return self._cs[item]

    def get_param(self, name: str) -> ParamType:
        return self._cs.get_hyperparameter(name)

    def get_params_dict(self) -> Dict[str, ParamType]:
        cs_dict = list(self._cs.get_hyperparameters_dict().items())
        if len(cs_dict) != len(self._cs_idx):
            raise RuntimeError('Inconsistent index mapping!')

        return {cs_dict[cs_idx][0]: cs_dict[cs_idx][1] for cs_idx in self._cs_idx}

    def get_params(self) -> List[ParamType]:
        return list(self.get_params_dict().values())

    def get_param_names(self) -> List[str]:
        return list(self.get_params_dict().keys())

    def get_param_by_idx(self, idx: int) -> str:
        return self._cs.get_hyperparameter_by_idx(self._cs_idx[idx])

    def get_idx_by_param_name(self, name: str) -> int:
        cs_idx = self._cs.get_idx_by_hyperparameter_name(name)
        return self._inv_cs_idx[cs_idx]

    def __str__(self):
        return f'Explicit design space:\n{self._cs!s}'

    def __repr__(self):
        return f'{self.__class__.__name__}; {self._cs!r}'

    def _block_after_init(self):
        if self._is_initialized:
            raise RuntimeError('Cannot change variables or constraints after usage!')

    def add_param(self, param: ParamType):
        self.add_params([param])

    def add_params(self, params: List[ParamType]):
        self._block_after_init()

        for param in params:
            if isinstance(param, Variable):
                raise ValueError('Parameters in the explicit design space are specified using '
                                 'FloatParam, IntParam or ChoiceParam')

        self._cs.add_hyperparameters(params)
        self._var_names += [param.name for param in params]
        self._update_cs_idx()

    def add_condition(self, condition: ConditionComponent):
        """Add a condition: https://automl.github.io/ConfigSpace/main/api/conditions.html"""
        self._block_after_init()

        self._cs.add_condition(condition)
        self._update_cs_idx()

    def add_conditions(self, conditions):
        """Add conditions: https://automl.github.io/ConfigSpace/main/api/conditions.html"""
        self._block_after_init()

        self._cs.add_conditions(conditions)
        self._update_cs_idx()

    def add_value_constraint(self, target_param: ParamType, target_value: Union[list, Any],
                             source_param: ParamType, source_value: Union[list, Any]):
        """Helper function to add a constraint (forbidden clause) preventing (any of) target_value on target_param to be
        selected if source_param has (one of) source_value"""

        target_clause = ForbiddenInClause(target_param, target_value) \
            if isinstance(target_value, Sequence) else ForbiddenEqualsClause(target_param, target_value)
        source_clause = ForbiddenInClause(source_param, source_value) \
            if isinstance(source_value, Sequence) else ForbiddenEqualsClause(source_param, source_value)

        self.add_forbidden_clause(ForbiddenAndConjunction(target_clause, source_clause))

    def add_forbidden_clause(self, clause: AbstractForbiddenComponent):
        """Add a forbidden clause: https://automl.github.io/ConfigSpace/main/api/forbidden_clauses.html"""
        self._block_after_init()

        self._cs.add_forbidden_clause(clause)
        self._update_cs_idx()

    def add_forbidden_clauses(self, clauses: List[AbstractForbiddenComponent]):
        """Add forbidden clauses: https://automl.github.io/ConfigSpace/main/api/forbidden_clauses.html"""
        self._block_after_init()

        self._cs.add_forbidden_clauses(clauses)
        self._update_cs_idx()

    def is_explicit(self) -> bool:
        return True

    def _get_variables(self) -> List[Variable]:
        """Returns the list of design variables (pymoo classes)"""
        des_vars = []
        for param in self.get_params():
            if isinstance(param, UniformFloatHyperparameter):
                des_vars.append(Real(bounds=(param.lower, param.upper)))

            elif isinstance(param, IntegerParam):
                des_vars.append(Integer(bounds=(param.lower, param.upper)))

            elif isinstance(param, CategoricalHyperparameter):
                des_vars.append(Choice(options=param.choices))

            else:
                raise ValueError(f'Unsupported parameter type: {param!r}')

        return des_vars

    def _is_conditionally_active(self) -> List[bool]:
        conditional_params = set(self._cs.get_all_conditional_hyperparameters())
        return [name in conditional_params for name in self.get_param_names()]

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        Imputation of inactive variables is handled automatically.
        """
        x_float = x.astype(float)
        self._cs_normalize_x(x_float)

        inv_cs_idx = self._inv_cs_idx
        configs = []
        for xi in x_float:
            configs.append(self._get_correct_config(xi[inv_cs_idx]))

        x[:, :], is_active[:, :] = self._configs_to_x(configs)

    def _get_correct_config(self, vector: np.ndarray) -> Configuration:
        config = Configuration(self._cs, vector=vector)

        # # Get active parameters and set values in the vector to NaN if they are inactive
        # x_active = self._cs.get_active_hyperparameters(config)
        # vector = config.get_array().copy()
        # is_inactive_mask = [name not in x_active for name in self._cs.get_hyperparameter_names()]
        # vector[is_inactive_mask] = np.nan
        #
        # # Check if the configuration also satisfies all forbidden clauses
        # config = Configuration(self._cs, vector=vector)
        # try:
        #     config.is_valid_configuration()
        # except (ValueError, ForbiddenValueError):
        #     # If not, create a random valid neighbor
        #     config = get_random_neighbor(config, seed=None)
        # return config

        # Unfortunately the above code doesn't work:
        # https://github.com/automl/ConfigSpace/issues/253#issuecomment-1513216665
        # Therefore, we temporarily fix it with a very dirty workaround: catch the error raised in check_configuration
        # to find out which parameters should be inactive
        while True:
            try:
                config.is_valid_configuration()
                return config

            except ValueError as e:
                error_str = str(e)
                if 'Inactive hyperparameter' in error_str:
                    # Deduce which parameter is inactive
                    inactive_param_name = error_str.split("'")[1]
                    param_idx = self._cs.get_idx_by_hyperparameter_name(inactive_param_name)

                    # Modify the vector and create a new Configuration
                    vector = config.get_array().copy()
                    vector[param_idx] = np.nan
                    config = Configuration(self._cs, vector=vector)

                # At this point, the parameter active statuses are set correctly, so we only need to correct the
                # configuration to one that does not violate the forbidden clauses
                elif isinstance(e, ForbiddenValueError):
                    return get_random_neighbor(config, seed=None)

                else:
                    raise

    def _quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n design vectors (also return is_active) without generating all design vectors first"""
        configs = self._cs.sample_configuration(n)
        if n == 1:
            configs = [configs]
        return self._configs_to_x(configs)

    def _get_n_valid_discrete(self) -> Optional[int]:
        """Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio"""
        # Currently only possible by generating all discrete x

    def _get_n_active_cont_mean(self) -> Optional[float]:
        """Currently only possible by generating all discrete x"""

    def _get_n_correct_discrete(self) -> Optional[int]:
        """Return the number of correct discrete design points (ignoring continuous dimensions); enables calculation of
        the correction ratio"""
        # Currently only possible by generating all discrete x

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        """Currently only possible by generating all discrete x"""

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate all possible discrete design vectors (if available). Returns design vectors and activeness
        information."""
        num_steps = {}
        for param in self.get_params():
            if isinstance(param, IntegerParam):
                num_steps[param.name] = param.upper-param.lower+1
            else:
                num_steps[param.name] = 1

        # Currently might not work if there are any forbidden clauses
        try:
            return self._configs_to_x(generate_grid(self._cs, num_steps))
        except (ForbiddenValueError, AssertionError):
            pass

        # Unfortunately there is a bug: generate_grid does not handle forbidden clauses
        cs_no_forbidden = NoDefaultConfigurationSpace(name='no_forbidden')
        cs_no_forbidden.add_hyperparameters(self._cs.get_hyperparameters())
        cs_no_forbidden.add_conditions(self._cs.get_conditions())

        configs_no_forbidden: List[Configuration] = generate_grid(cs_no_forbidden, num_steps)

        # Filter out configs that violate the forbidden clauses
        configs = []
        for config_no_forbidden in configs_no_forbidden:
            try:
                config = Configuration(self._cs, values=config_no_forbidden.get_dictionary())
            except ForbiddenValueError:
                continue
            configs.append(config)

        return self._configs_to_x(configs)

    def _configs_to_x(self, configs: List[Configuration]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(configs), self.n_var))
        is_active = np.zeros((len(configs), self.n_var), dtype=bool)
        if len(configs) == 0:
            return x, is_active

        cs_idx = self._cs_idx
        for i, config in enumerate(configs):
            x[i, :] = config.get_array()[cs_idx]

        # De-normalize continuous and integer variables
        self._cs_denormalize_x(x)

        is_active = np.isfinite(x)
        x[~is_active] = 0
        return x, is_active

    def _cs_normalize_x(self, x: np.ndarray):
        xl, xu = self.xl, self.xu
        norm = xu-xl
        norm[norm == 0] = 1e-16

        is_cont_mask, is_int_mask = self.is_cont_mask, self.is_int_mask
        x[:, is_cont_mask] = np.clip((x[:, is_cont_mask]-xl[is_cont_mask])/norm[is_cont_mask], 0, 1)

        # Integer values are normalized similarly to what we do in round_x_discrete
        x[:, is_int_mask] = (x[:, is_int_mask]-xl[is_int_mask]+.49999)/(norm[is_int_mask]+.9999)

    def _cs_denormalize_x(self, x: np.ndarray):
        xl, xu = self.xl, self.xu
        is_cont_mask, is_int_mask = self.is_cont_mask, self.is_int_mask
        x[:, is_cont_mask] = x[:, is_cont_mask]*(xu[is_cont_mask]-xl[is_cont_mask])+xl[is_cont_mask]

        # Integer values are normalized similarly to what we do in round_x_discrete
        x[:, is_int_mask] = np.round(x[:, is_int_mask]*(xu[is_int_mask]-xl[is_int_mask]+.9999)+xl[is_int_mask]-.49999)


class NoDefaultConfigurationSpace(ConfigurationSpace):
    """ConfigurationSpace that supports no default configuration"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if kwargs.get('seed') is None:
            self.random = get_np_random_singleton()

    def get_default_configuration(self, *args, **kwargs):
        raise NotImplementedError

    def _check_default_configuration(self, *args, **kwargs):
        pass
