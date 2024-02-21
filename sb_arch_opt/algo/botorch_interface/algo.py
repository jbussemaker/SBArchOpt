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
import pymoo.core.variable as var
from pymoo.core.population import Population
from sb_arch_opt.problem import ArchOptProblemBase

try:
    from ax import ParameterType, RangeParameter, ChoiceParameter, SearchSpace, Experiment, OptimizationConfig, \
        Objective, MultiObjective, OutcomeConstraint, Metric, ComparisonOp, MultiObjectiveOptimizationConfig, \
        Trial, Data
    from ax.service.managed_loop import OptimizationLoop
    from ax.modelbridge.dispatch_utils import choose_generation_strategy

    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False

__all__ = ['AxInterface', 'check_dependencies']


def check_dependencies():
    if not HAS_BOTORCH:
        raise ImportError('BoTorch/Ax dependencies not installed: python setup.py install[botorch]')


class AxInterface:
    """
    Class handling interfacing between ArchOptProblemBase and the Ax optimization loop, based on:
    https://ax.dev/tutorials/gpei_hartmann_developer.html

    Restart can be implemented based on:
    - https://ax.dev/tutorials/generation_strategy.html#3B.-JSON-storage
    - https://ax.dev/tutorials/gpei_hartmann_service.html#7.-Save-/-reload-optimization-to-JSON-/-SQL
    Failed trails can be marked (as a primitive way of dealing with hidden constraints):
    - https://ax.dev/tutorials/gpei_hartmann_service.html#Special-Cases
    """

    def __init__(self, problem: ArchOptProblemBase):
        check_dependencies()
        self._problem = problem

    def get_optimization_loop(self, n_init: int, n_infill: int, seed: int = None) -> 'OptimizationLoop':
        experiment = self.get_experiment()
        n_eval_total = n_init+n_infill
        generation_strategy = choose_generation_strategy(
            search_space=experiment.search_space,
            experiment=experiment,
            num_trials=n_eval_total,
            num_initialization_trials=n_init,
            max_parallelism_override=self._problem.get_n_batch_evaluate(),
        )

        return OptimizationLoop(
            experiment=experiment,
            evaluation_function=self.evaluate,
            total_trials=n_eval_total,
            generation_strategy=generation_strategy,
            random_seed=seed,
        )

    def get_search_space(self) -> 'SearchSpace':
        """Gets the search space as defined by the underlying problem"""
        parameters = []
        for i, var_def in enumerate(self._problem.des_vars):
            name = f'x{i}'
            if isinstance(var_def, var.Real):
                parameters.append(RangeParameter(
                    name=name, parameter_type=ParameterType.FLOAT, lower=var_def.bounds[0], upper=var_def.bounds[1]))

            elif isinstance(var_def, var.Integer):
                parameters.append(RangeParameter(
                    name=name, parameter_type=ParameterType.INT, lower=var_def.bounds[0], upper=var_def.bounds[1]))

            elif isinstance(var_def, var.Binary):
                parameters.append(ChoiceParameter(
                    name=name, parameter_type=ParameterType.INT, values=[0, 1], is_ordered=True))

            elif isinstance(var_def, var.Choice):
                parameters.append(ChoiceParameter(
                    name=name, parameter_type=ParameterType.INT, values=var_def.options, is_ordered=False))

            else:
                raise RuntimeError(f'Unsupported design variable type: {var_def!r}')

        return SearchSpace(parameters)

    def get_optimization_config(self) -> 'OptimizationConfig':
        """Gets the optimization config (objectives and constraints) as defined by the underlying problem"""

        if self._problem.n_eq_constr > 0:
            raise RuntimeError('Currently equality constraints are not supported!')
        constraints = [OutcomeConstraint(Metric(name=f'g{i}'), ComparisonOp.LEQ, bound=0., relative=False)
                       for i in range(self._problem.n_ieq_constr)]

        if self._problem.n_obj == 1:
            return OptimizationConfig(
                objective=Objective(Metric(name='f0'), minimize=True),
                outcome_constraints=constraints,
            )

        objective = MultiObjective(objectives=[
            Objective(Metric(name=f'f{i}'), minimize=True) for i in range(self._problem.n_obj)])

        return MultiObjectiveOptimizationConfig(
            objective=objective,
            outcome_constraints=constraints,
        )

    def get_experiment(self) -> 'Experiment':
        return Experiment(
            name=repr(self._problem),
            search_space=self.get_search_space(),
            optimization_config=self.get_optimization_config(),
        )

    def evaluate(self, parameterization: dict, _=None) -> dict:
        x = np.array([[parameterization[f'x{i}'] for i in range(self._problem.n_var)]])
        out = self._problem.evaluate(x, return_as_dictionary=True)

        metrics = {}
        for i in range(self._problem.n_obj):
            metrics[f'f{i}'] = out['F'][0, i]
        for i in range(self._problem.n_ieq_constr):
            metrics[f'g{i}'] = out['G'][0, i]
        return metrics

    def get_population(self, opt_loop: 'OptimizationLoop') -> Population:
        x, f, g = [], [], []
        data_by_trial = opt_loop.experiment.data_by_trial
        trial: 'Trial'
        for trial in opt_loop.experiment.trials.values():
            x.append([trial.arm.parameters[f'x{i}'] for i in range(self._problem.n_var)])

            data: 'Data' = list(data_by_trial[trial.index].values())[0]
            values = data.df.set_index('metric_name')['mean']
            f.append([values[f'f{i}'] for i in range(self._problem.n_obj)])
            g.append([values[f'g{i}'] for i in range(self._problem.n_ieq_constr)])

        kwargs = {'X': np.array(x), 'F': np.array(f)}
        if self._problem.n_ieq_constr > 0:
            kwargs['G'] = np.array(g)
        return Population.new(**kwargs)
