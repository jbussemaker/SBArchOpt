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
import os
import copy
import numpy as np
from typing import *
from dataclasses import dataclass
import pymoo.core.variable as var
from pymoo.core.problem import Problem
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.design_space import ArchDesignSpace
from sb_arch_opt.sampling import HierarchicalSampling
from pymoo.util.normalization import Normalization, SimpleZeroToOneNormalization
HAS_ARCH_SBO = True
HAS_SMT = True

try:
    os.environ['USE_NUMBA_JIT'] = '1'
    from smt.surrogate_models.surrogate_model import SurrogateModel

    from smt.surrogate_models.krg import KRG, KrgBased
    from smt.surrogate_models.kpls import KPLS
    from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType

    from smt.utils.design_space import BaseDesignSpace
    import smt.utils.design_space as ds

    from smt import __version__

except ImportError:
    HAS_ARCH_SBO = False
    HAS_SMT = False

    class BaseDesignSpace:
        pass

    class SurrogateModel:
        pass

__all__ = ['check_dependencies', 'HAS_ARCH_SBO', 'HAS_SMT', 'ModelFactory', 'MixedDiscreteNormalization', 'SBArchOptDesignSpace',
           'MultiSurrogateModel']


def check_dependencies():
    if not HAS_ARCH_SBO:
        raise ImportError('ArchSBO dependencies not installed: pip install sb-arch-opt[arch_sbo]')


@dataclass
class SMTDesignSpaceSpec:
    var_defs: List[dict]  # [{'name': name, 'lb': lb, 'ub', ub}, ...]
    design_space: 'SBArchOptDesignSpace'
    is_mixed_discrete: bool


class MixedDiscreteNormalization(Normalization):
    """Normalizes continuous variables to [0, 1], moves integer variables to start at 0"""

    def __init__(self, design_space: ArchDesignSpace):
        self._design_space = design_space
        self._is_cont_mask = design_space.is_cont_mask
        self._is_int_mask = design_space.is_int_mask
        super().__init__()

    def forward(self, x):
        x_norm = x.copy()
        xl, xu = self._design_space.xl, self._design_space.xu

        norm = xu - xl
        norm[norm == 0] = 1e-32

        cont_mask = self._is_cont_mask
        x_norm[:, cont_mask] = (x[:, cont_mask] - xl[cont_mask]) / norm[cont_mask]

        int_mask = self._is_int_mask
        x_norm[:, int_mask] = x[:, int_mask] - xl[int_mask]

        return x_norm

    def backward(self, x):
        x_abs = x.copy()
        xl, xu = self._design_space.xl, self._design_space.xu

        cont_mask = self._is_cont_mask
        x_abs[:, cont_mask] = x[:, cont_mask]*(xu[cont_mask]-xl[cont_mask]) + xl[cont_mask]

        int_mask = self._is_int_mask
        x_abs[:, int_mask] = x[:, int_mask] + xl[int_mask]

        return x_abs


class ModelFactory:

    def __init__(self, problem: ArchOptProblemBase):
        self.problem = problem

    def get_smt_design_space_spec(self) -> SMTDesignSpaceSpec:
        """Get information about the design space as needed by SMT and SEGOMOE"""
        check_dependencies()
        return self.create_smt_design_space_spec(self.problem.design_space)

    @staticmethod
    def create_smt_design_space_spec(arch_design_space: ArchDesignSpace, md_normalize=False, cont_relax=False,
                                     ignore_hierarchy=False):
        check_dependencies()

        design_space = SBArchOptDesignSpace(arch_design_space, md_normalize=md_normalize, cont_relax=cont_relax,
                                            ignore_hierarchy=ignore_hierarchy)
        is_mixed_discrete = not np.all(arch_design_space.is_cont_mask)

        var_defs = [{'name': f'x{i}', 'lb': bounds[0], 'ub': bounds[1]}
                    for i, bounds in enumerate(design_space.get_num_bounds())]

        return SMTDesignSpaceSpec(
            var_defs=var_defs,
            design_space=design_space,
            is_mixed_discrete=is_mixed_discrete,
        )

    @staticmethod
    def get_continuous_normalization(problem: Problem):
        return SimpleZeroToOneNormalization(xl=problem.xl, xu=problem.xu, estimate_bounds=False)

    def get_md_normalization(self):
        return MixedDiscreteNormalization(self.problem.design_space)

    @staticmethod
    def get_rbf_model():
        check_dependencies()
        from smt.surrogate_models.rbf import RBF
        return RBF(print_global=False, d0=1., poly_degree=-1, reg=1e-10)

    @staticmethod
    def get_kriging_model(multi=True, kpls_n_comp: int = None, **kwargs):
        check_dependencies()

        if kpls_n_comp is not None:
            surrogate = KPLS(print_global=False, n_comp=kpls_n_comp, **kwargs)
        else:
            surrogate = KRG(print_global=False, **kwargs)

        if multi:
            surrogate = MultiSurrogateModel(surrogate)
        return surrogate

    def get_md_kriging_model(self, kpls_n_comp: int = None, multi=True, ignore_hierarchy=False,
                             **kwargs_) -> Tuple['SurrogateModel', Normalization]:
        check_dependencies()
        normalization = self.get_md_normalization()
        design_space = self.problem.design_space
        norm_ds_spec = self.create_smt_design_space_spec(
            design_space, md_normalize=True, ignore_hierarchy=ignore_hierarchy)

        kwargs = dict(
            print_global=False,
            design_space=norm_ds_spec.design_space,
            categorical_kernel=MixIntKernelType.GOWER,
            hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
        )
        kwargs.update(kwargs_)

        # Disable KPLS if the nr of requested components is too high
        if kpls_n_comp is not None:
            n_dim_apply_pls = design_space.n_var
            
            if kpls_n_comp > n_dim_apply_pls:
                kpls_n_comp = None

        if kpls_n_comp is not None:
            kwargs['categorical_kernel'] = MixIntKernelType.CONT_RELAX

            # Ignore hierarchy in the design space as KPLS does not support this
            non_hier_ds_spec = self.create_smt_design_space_spec(
                self.problem.design_space, md_normalize=True, ignore_hierarchy=True)
            kwargs['design_space'] = non_hier_ds_spec.design_space

            surrogate = KPLS(n_comp=kpls_n_comp, **kwargs)
        else:
            surrogate = KRG(**kwargs)

        if ignore_hierarchy or kpls_n_comp is not None:
            surrogate.supports['x_hierarchy'] = False

        if multi:
            surrogate = MultiSurrogateModel(surrogate)

        return surrogate, normalization

    @staticmethod
    def get_n_theta(problem: ArchOptProblemBase, surrogate: 'SurrogateModel') -> int:

        def _get_n_theta(model: 'SurrogateModel') -> int:
            if isinstance(model, KrgBased):
                if hasattr(model, 'optimal_theta') and len(model.optimal_theta):
                    return len(model.optimal_theta)

                n_train = 2
                if isinstance(model, KPLS):
                    n_train = model.options['n_comp']+1
                n_theta = 0

                def _override(theta):
                    nonlocal n_theta
                    # No need to actually train the model: we only want to know how many hyperparams we have
                    n_theta = len(theta)
                    raise RuntimeError

                model = copy.deepcopy(model)
                model.options['n_start'] = 1
                model.set_training_values(np.zeros((n_train, problem.n_var)), np.zeros((n_train, 1)))
                model._reduced_likelihood_function = _override
                try:
                    model.train()
                except RuntimeError:
                    pass
                return n_theta

            raise RuntimeError(f'Not a Kriging model: {surrogate!r}')

        if isinstance(surrogate, MultiSurrogateModel):
            if len(surrogate._models) == 0:
                n_single = _get_n_theta(surrogate._surrogate)
            else:
                n_single = _get_n_theta(surrogate._models[0])

            ny = problem.n_obj + problem.n_ieq_constr
            return n_single * ny

        return _get_n_theta(surrogate)


class SBArchOptDesignSpace(BaseDesignSpace):
    """SMT design space implementation using SBArchOpt's design space logic"""
    def __init__(self, arch_design_space: ArchDesignSpace, md_normalize=False, cont_relax=False,
                 ignore_hierarchy=False):
        self._ds = arch_design_space
        self.normalize = MixedDiscreteNormalization(arch_design_space) if md_normalize else None
        self._cont_relax = cont_relax
        self._ignore_hierarchy = ignore_hierarchy
        super().__init__()

    @property
    def arch_design_space(self) -> ArchDesignSpace:
        return self._ds

    def _get_design_variables(self) -> List['ds.DesignVariable']:
        """Return the design variables defined in this design space if not provided upon initialization of the class"""
        smt_des_vars = []
        is_dv_cond = ([False]*len(self._ds.des_vars)) if self._ignore_hierarchy else self._ds.is_conditionally_active
        normalize = self.normalize is not None
        cont_relax = self._cont_relax
        for i, dv in enumerate(self._ds.des_vars):
            if isinstance(dv, var.Real):
                bounds = (0, 1) if normalize else dv.bounds
                smt_des_vars.append(ds.FloatVariable(bounds[0], bounds[1]))

            elif isinstance(dv, var.Integer):
                bounds = (0, dv.bounds[1]-dv.bounds[0]) if normalize else dv.bounds
                if cont_relax:
                    smt_des_vars.append(ds.FloatVariable(bounds[0], bounds[1]))
                else:
                    smt_des_vars.append(ds.IntegerVariable(bounds[0], bounds[1]))

            elif isinstance(dv, var.Binary):
                if cont_relax:
                    smt_des_vars.append(ds.FloatVariable(0, 1))
                else:
                    smt_des_vars.append(ds.OrdinalVariable(values=[0, 1]))

            elif isinstance(dv, var.Choice):
                if cont_relax:
                    smt_des_vars.append(ds.FloatVariable(0, len(dv.options)-1))
                else:
                    smt_des_vars.append(ds.CategoricalVariable(values=dv.options))

            else:
                raise ValueError(f'Unexpected variable type: {dv!r}')
        return smt_des_vars

    def _is_conditionally_acting(self) -> np.ndarray:
        if self._ignore_hierarchy:
            return np.zeros((len(self._ds.des_vars),), dtype=bool)

        return self._ds.is_conditionally_active

    def _correct_get_acting(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.normalize is not None:
            x = self.normalize.backward(x)

        x, is_active = self._ds.correct_x(x)

        if self.normalize is not None:
            x = self.normalize.forward(x)

        if self._ignore_hierarchy:
            is_active = np.ones(is_active.shape, dtype=bool)
        return x, is_active

    def _sample_valid_x(self, n: int, random_state=None) -> Tuple[np.ndarray, np.ndarray]:
        sampler = HierarchicalSampling(seed=random_state)
        stub_problem = ArchOptProblemBase(self._ds)
        x, is_active = sampler.sample_get_x(stub_problem, n)

        if self.normalize is not None:
            x = self.normalize.forward(x)

        if self._ignore_hierarchy:
            is_active = np.ones(is_active.shape, dtype=bool)
        return x, is_active

    def __str__(self):
        return 'SBArchOpt Design Space'

    def __repr__(self):
        return f'{self.__class__.__name__}({self._ds!r})'


class MultiSurrogateModel(SurrogateModel):
    """SMT surrogate model wrapper that trains independent models for each provided output"""

    def __init__(self, surrogate: 'SurrogateModel', **kwargs):
        super().__init__(**kwargs)

        self._surrogate = surrogate
        self._is_krg = isinstance(surrogate, KrgBased)
        self._models: List['SurrogateModel'] = []
        self.supports = self._surrogate.supports
        self.options["print_global"] = False

        self.xt = None
        self.yt = None

    @property
    def name(self):
        return f'Multi{self._surrogate.name}'

    def _initialize(self):
        self.supports["derivatives"] = False

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None, is_acting=None) -> None:
        self.xt = xt
        self.yt = yt

        self._models = models = []
        for iy in range(yt.shape[1]):
            model: Union['KrgBased', 'SurrogateModel'] = self._copy_underlying()
            if self._is_krg:
                model.set_training_values(xt, yt[:, [iy]], is_acting=is_acting)
            else:
                model.set_training_values(xt, yt[:, [iy]])
            models.append(model)

    def _copy_underlying(self) -> 'SurrogateModel':
        model = self._surrogate

        design_space = None
        has_ds = 'design_space' in model.options
        if has_ds:
            design_space = model.options['design_space']
            if design_space is not None:
                model.options['design_space'] = []

        model_copy = copy.deepcopy(model)

        if has_ds and design_space is not None:
            model.options['design_space'] = design_space
            model_copy.options['design_space'] = design_space
        return model_copy

    def train(self) -> None:
        theta0 = None
        for i, model in enumerate(self._models):
            if i > 0 and isinstance(model, KrgBased) and theta0 is not None:
                model.options['theta0'] = theta0

            model.train()

            # rmse = np.linalg.norm(self.yt[:, i] - model.predict_values(self.xt)[:, 0], 2)
            # print(f'TRAINED {i}: {rmse:.3g}')

            if i == 0 and isinstance(model, KrgBased):
                try:
                    theta0 = list(model.optimal_theta)
                except AttributeError:
                    pass

    def predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        model: Union['SurrogateModel', 'KrgBased']
        if self._is_krg:
            values = [model.predict_values(x, is_acting=is_acting) for model in self._models]
        else:
            values = [model.predict_values(x) for model in self._models]
        return np.column_stack(values)

    def predict_variances(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        model: Union['SurrogateModel', 'KrgBased']
        if self._is_krg:
            values = [model.predict_variances(x, is_acting=is_acting) for model in self._models]
        else:
            values = [model.predict_variances(x) for model in self._models]
        return np.column_stack(values)

    def _predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        raise RuntimeError
