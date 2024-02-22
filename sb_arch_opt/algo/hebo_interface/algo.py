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
import logging
import numpy as np
import pandas as pd
from typing import *
import pymoo.core.variable as var
from pymoo.core.population import Population
from sb_arch_opt.problem import ArchOptProblemBase

try:
    from hebo.design_space.param import Parameter
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    from hebo.optimizers.general import GeneralBO
    from hebo.optimizers.abstract_optimizer import AbstractOptimizer

    HAS_HEBO = True
except ImportError:
    HAS_HEBO = False

__all__ = ['HAS_HEBO', 'check_dependencies', 'HEBOArchOptInterface']

log = logging.getLogger('sb_arch_opt.hebo')


def check_dependencies():
    if not HAS_HEBO:
        raise ImportError('HEBO dependencies not installed: python setup.py install[hebo]')


class HEBOArchOptInterface:
    """
    Interface class to HEBO algorithm.
    """

    def __init__(self, problem: ArchOptProblemBase, n_init: int, seed: int = None):
        check_dependencies()
        self._problem = problem
        self._n_init = n_init
        self._optimizer = None
        self._design_space = None

        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

    @property
    def problem(self):
        return self._problem

    @property
    def n_batch(self):
        n_batch = self._problem.get_n_batch_evaluate()
        return n_batch if n_batch is not None else 1

    @property
    def design_space(self) -> 'DesignSpace':
        if self._design_space is None:
            hebo_var_defs = []
            for i, var_def in enumerate(self._problem.des_vars):
                name = f'x{i}'

                if isinstance(var_def, var.Real):
                    hebo_var_defs.append({'name': name, 'type': 'num', 'lb': var_def.bounds[0], 'ub': var_def.bounds[1]})

                elif isinstance(var_def, var.Integer):
                    hebo_var_defs.append({'name': name, 'type': 'int', 'lb': var_def.bounds[0], 'ub': var_def.bounds[1]})

                elif isinstance(var_def, var.Binary):
                    hebo_var_defs.append({'name': name, 'type': 'bool'})

                elif isinstance(var_def, var.Choice):
                    hebo_var_defs.append({'name': name, 'type': 'cat', 'categories': var_def.options})

                else:
                    raise RuntimeError(f'Unsupported design variable type: {var_def!r}')

            self._design_space = DesignSpace().parse(hebo_var_defs)
        return self._design_space

    @property
    def optimizer(self) -> 'AbstractOptimizer':
        if self._optimizer is None:
            if self._problem.n_obj == 1 and self._problem.n_ieq_constr == 0:
                self._optimizer = HEBO(self.design_space, model_name='gpy', rand_sample=self._n_init,
                                       scramble_seed=self._seed)
            else:
                self._optimizer = GeneralBO(self.design_space, num_obj=self._problem.n_obj,
                                            num_constr=self._problem.n_ieq_constr, rand_sample=self._n_init,
                                            model_config={'num_epochs': 100})
        return self._optimizer

    @property
    def pop(self) -> Population:
        x = self._to_x(self.optimizer.X)

        y: np.ndarray = self.optimizer.y
        f = y[:, :self._problem.n_obj]
        kwargs = {'X': x, 'F': f}
        if self._problem.n_ieq_constr > 0:
            kwargs['G'] = y[:, self._problem.n_obj:]

        return Population.new(**kwargs)

    def optimize(self, n_infill: int):
        """Run the optimization loop for n_infill infill points (on top on the initialization points)"""
        n_total = self._n_init+n_infill
        evaluated = 0
        while evaluated < n_total:
            x = self.ask()

            out = self._problem.evaluate(x, return_as_dictionary=True)
            x_eval = out['X']
            f = out['F']
            g = out['G'] if self._problem.n_ieq_constr > 0 else None

            self.tell(x_eval, f, g)
            evaluated += x_eval.shape[0]

    def ask(self) -> np.ndarray:
        """Returns n_batch infill points"""
        x_df = self.optimizer.suggest(n_suggestions=self.n_batch)
        return self._to_x(x_df)

    def tell(self, x: np.ndarray, f: np.ndarray, g: np.ndarray = None):
        """Updates optimizer with evaluated design points"""
        y = f
        if g is not None:
            y = np.column_stack([f, g])

        params: List['Parameter'] = self.design_space.paras.values()
        x_df = pd.DataFrame({f'x{i}': param.inverse_transform(x[:, i]) for i, param in enumerate(params)})

        self.optimizer.observe(x_df, y)

    def _to_x(self, x_df: pd.DataFrame) -> np.ndarray:
        params: List['Parameter'] = self.design_space.paras.values()
        return np.column_stack([param.transform(x_df[f'x{i}']) for i, param in enumerate(params)])
