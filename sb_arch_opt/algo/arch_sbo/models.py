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
from typing import *
from dataclasses import dataclass
import pymoo.core.variable as var
from pymoo.core.problem import Problem
from sb_arch_opt.problem import ArchOptProblemBase
from pymoo.util.normalization import Normalization, SimpleZeroToOneNormalization

try:
    from smt.surrogate_models.rbf import RBF
    from smt.surrogate_models.krg import KRG
    from smt.applications.mixed_integer import MixedIntegerSurrogateModel
    from smt.surrogate_models.surrogate_model import SurrogateModel
    from smt.applications.moe import MOESurrogateModel

    try:
        # SMT v1.3
        from enum import Enum
        from smt.applications.mixed_integer import FLOAT as float_type, INT as int_type, ENUM as enum_type

        class XType:
            FLOAT = float_type
            ORD = int_type
            ENUM = enum_type

        class XRole:
            NEUTRAL = 'NEUTRAL'
            META = 'META'
            DECREED = 'DECREED'

        IS_SMT_V2 = False

        class XSpecs:
            def __init__(self, **kwargs):
                pass

    except ImportError:
        # Temp fix: fix class name of XRole enum
        from enum import Enum
        import smt.utils.kriging as krg_utils
        krg_utils.XRole = Enum("XRole", ["NEUTRAL", "META", "DECREED"])

        # SMT v2
        from smt.utils.mixed_integer import XType
        from smt.utils.kriging import XSpecs, XRole
        IS_SMT_V2 = True

    HAS_ARCH_SBO = True
except ImportError:
    HAS_ARCH_SBO = False
    get_sbo_termination = lambda *_, **__: None

__all__ = ['check_dependencies', 'HAS_ARCH_SBO', 'ModelFactory', 'MixedDiscreteNormalization']


def check_dependencies():
    if not HAS_ARCH_SBO:
        raise ImportError(f'arch_sbo dependencies not installed: python setup.py install[arch_sbo]')


@dataclass
class SMTDesignSpaceSpec:
    var_defs: List[dict]  # [{'name': name, 'lb': lb, 'ub', ub}, ...]
    var_types: List[Union[str, Tuple[str, int]]]  # FLOAT, INT, ENUM
    var_limits: List[Union[Tuple[float, float], list]]  # Bounds (options for an enum)
    x_specs: XSpecs
    is_mixed_discrete: bool


class MixedDiscreteNormalization(Normalization):
    """Normalizes continuous variables to [0, 1], moves integer variables to start at 0"""

    def __init__(self, problem: ArchOptProblemBase):
        self._problem = problem
        self._is_cont_mask = problem.is_cont_mask
        self._is_int_mask = problem.is_int_mask
        super().__init__()

    def forward(self, x):
        x_norm = x.copy()
        xl, xu = self._problem.xl, self._problem.xu

        norm = xu - xl
        norm[norm == 0] = 1e-32

        cont_mask = self._is_cont_mask
        x_norm[:, cont_mask] = (x[:, cont_mask] - xl[cont_mask]) / norm[cont_mask]

        int_mask = self._is_int_mask
        x_norm[:, int_mask] = x[:, int_mask] - xl[int_mask]

        return x_norm

    def backward(self, x):
        x_abs = x.copy()
        xl, xu = self._problem.xl, self._problem.xu

        cont_mask = self._is_cont_mask
        x_abs[:, cont_mask] = x[:, cont_mask]*(xu[cont_mask]-xl[cont_mask]) + xl[cont_mask]

        int_mask = self._is_int_mask
        x_abs[:, int_mask] = x[:, int_mask] + xl[int_mask]

        return x_abs

    @staticmethod
    def normalize_design_space(ds_spec: SMTDesignSpaceSpec) -> SMTDesignSpaceSpec:
        norm_var_defs = []
        norm_var_limits = []
        for i, var_limits in enumerate(ds_spec.var_limits):
            var_def = ds_spec.var_defs[i]
            if ds_spec.var_types[i] == XType.FLOAT:
                norm_var_limits.append((0, 1))
                norm_var_defs.append({**var_def, **{'lb': 0, 'ub': 1}})

            elif ds_spec.var_types[i] == XType.ORD:
                xu = var_limits[1]-var_limits[0]
                norm_var_limits.append((0, xu))
                norm_var_defs.append({**var_def, **{'lb': 0, 'ub': xu}})

            else:
                norm_var_limits.append(var_limits)
                norm_var_defs.append(var_def)

        return SMTDesignSpaceSpec(
            var_defs=norm_var_defs,
            var_types=ds_spec.var_types,
            var_limits=norm_var_limits,
            x_specs=XSpecs(
                xtypes=ds_spec.var_types,
                xlimits=norm_var_limits,
                xroles=[XRole.NEUTRAL]*len(norm_var_limits),
            ),
            is_mixed_discrete=ds_spec.is_mixed_discrete,
        )


class ModelFactory:

    def __init__(self, problem: ArchOptProblemBase):
        self.problem = problem

    def get_smt_design_space_spec(self) -> SMTDesignSpaceSpec:
        """Get information about the design space as needed by SMT and SEGOMOE"""
        check_dependencies()
        var_defs = []
        var_types = []
        var_limits = []
        is_mixed_discrete = False
        xl, xu = self.problem.xl, self.problem.xu
        for i, var_def in enumerate(self.problem.des_vars):
            name = f'x{i}'
            var_defs.append({'name': name, 'lb': xl[i], 'ub': xu[i]})

            if isinstance(var_def, var.Real):
                var_types.append(XType.FLOAT)
                var_limits.append(var_def.bounds)

            elif isinstance(var_def, var.Integer):
                is_mixed_discrete = True
                var_types.append(XType.ORD)
                var_limits.append(var_def.bounds)

            elif isinstance(var_def, var.Binary):
                is_mixed_discrete = True
                var_types.append(XType.ORD)
                var_limits.append([0, 1])

            elif isinstance(var_def, var.Choice):
                is_mixed_discrete = True
                var_types.append((XType.ENUM, len(var_def.options)))
                var_limits.append(list(range(len(var_def.options))))

            else:
                raise RuntimeError(f'Unsupported design variable type: {var_def!r}')

        x_specs = XSpecs(
            xtypes=var_types,
            xlimits=var_limits,
            xroles=[XRole.NEUTRAL]*len(var_types),
        )

        return SMTDesignSpaceSpec(
            var_defs=var_defs,
            var_types=var_types,
            var_limits=var_limits,
            x_specs=x_specs,
            is_mixed_discrete=is_mixed_discrete,
        )

    @staticmethod
    def get_continuous_normalization(problem: Problem):
        return SimpleZeroToOneNormalization(xl=problem.xl, xu=problem.xu, estimate_bounds=False)

    def get_md_normalization(self):
        return MixedDiscreteNormalization(self.problem)

    @staticmethod
    def get_rbf_model():
        check_dependencies()
        return RBF(print_global=False, d0=1., poly_degree=-1, reg=1e-10)

    @staticmethod
    def get_kriging_model(**kwargs):
        check_dependencies()
        return KRG(print_global=False, **kwargs)

    def get_md_kriging_model(self, **kwargs) -> Tuple['SurrogateModel', Normalization]:
        check_dependencies()
        normalization = self.get_md_normalization()
        norm_ds_spec = normalization.normalize_design_space(self.get_smt_design_space_spec())

        if norm_ds_spec.is_mixed_discrete:
            kwargs['n_start'] = kwargs.get('n_start', 5)

        if IS_SMT_V2:
            from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType
            from smt.applications.mixed_integer import MixedIntegerKrigingModel
            surrogate = KRG(
                print_global=False,
                xspecs=norm_ds_spec.x_specs,
                categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
                hierarchical_kernel=MixHrcKernelType.ALG_KERNEL,
                **kwargs,
            )
            if norm_ds_spec.is_mixed_discrete:
                surrogate = MixedIntegerKrigingModel(surrogate=surrogate)
            return surrogate, normalization

        from smt.applications.mixed_integer import MixedIntegerSurrogateModel
        surrogate = KRG(print_global=False, **kwargs)
        if norm_ds_spec.is_mixed_discrete:
            surrogate = MixedIntegerSurrogateModel(
                norm_ds_spec.var_types, norm_ds_spec.var_limits,
                surrogate, input_in_folded_space=True,
            )
        return surrogate, normalization

    @staticmethod
    def get_moe_model(**kwargs):
        check_dependencies()
        kwargs['allow'] = kwargs.get('allow', ['KRG', 'KPLS', 'LS', 'IDW', 'RBF', 'QP'])
        return MOESurrogateModel(print_global=False, **kwargs)
