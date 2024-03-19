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
import logging
import pathlib
import numpy as np
from typing import *
import pymoo.core.variable as var
from sb_arch_opt.util import capture_log
from pymoo.core.population import Population
from sb_arch_opt.problem import ArchOptProblemBase

# https://github.com/explosion/spaCy/issues/7664#issuecomment-825501808
# Needed to solve "Fatal Python error: aborted"!
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

try:
    from trieste.bayesian_optimizer import BayesianOptimizer, TrainableProbabilisticModelType, FrozenRecord, Record
    from trieste.observer import Observer, MultiObserver, Dataset, OBJECTIVE
    from trieste.space import SearchSpace, Box, DiscreteSearchSpace, TaggedProductSearchSpace

    from trieste.models.gpflow import build_gpr, GaussianProcessRegression, build_vgp_classifier, \
        VariationalGaussianProcess
    from trieste.models.optimizer import BatchOptimizer
    from trieste.models.interfaces import TrainableProbabilisticModel, ProbabilisticModel
    from trieste.acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
    from trieste.acquisition import ProbabilityOfFeasibility, ExpectedConstrainedImprovement, \
        ExpectedHypervolumeImprovement, ExpectedConstrainedHypervolumeImprovement, ExpectedImprovement, Product, \
        SingleModelAcquisitionBuilder

    import tensorflow as tf
    from dill import UnpicklingError

    HAS_TRIESTE = True
except ImportError:
    class BayesianOptimizer:
        pass

    class SingleModelAcquisitionBuilder:
        pass

    HAS_TRIESTE = False
    OBJECTIVE = 'OBJECTIVE'

__all__ = ['HAS_TRIESTE', 'check_dependencies', 'ArchOptBayesianOptimizer', 'OBJECTIVE', 'CONSTR_PREFIX',
           'ProbabilityOfValidity']

log = logging.getLogger('sb_arch_opt.trieste')

CONSTR_PREFIX = 'G'
FAILED = 'FAILED'


def check_dependencies():
    if not HAS_TRIESTE:
        raise ImportError('Trieste dependencies not installed: python setup.py install[trieste]')


class ArchOptBayesianOptimizer(BayesianOptimizer):
    """
    Bayesian optimization loop controller with some extra helper functions.
    Use the `run_optimization` function to run the DOE and infill loops.
    Use `initialize_from_previous` to initialize the optimization state from previously saved results.

    Optimization loop: https://secondmind-labs.github.io/trieste/2.0.0/notebooks/expected_improvement.html
    Restart: https://secondmind-labs.github.io/trieste/2.0.0/notebooks/recovering_from_errors.html
    Constraints: https://secondmind-labs.github.io/trieste/2.0.0/notebooks/inequality_constraints.html
    Multi-objective: https://secondmind-labs.github.io/trieste/2.0.0/notebooks/multi_objective_ehvi.html
    Hidden constraints: https://secondmind-labs.github.io/trieste/2.0.0/notebooks/failure_ego.html
    Ask-tell: https://secondmind-labs.github.io/trieste/2.0.0/notebooks/ask_tell_optimization.html
    """

    def __init__(self, problem: ArchOptProblemBase, n_init: int, n_infill: int, pof=.5,
                 rule: 'AcquisitionRule' = None, seed: int = None):
        check_dependencies()
        self._problem = problem
        self.pof = pof
        self._rule = rule
        self.n_init = n_init
        self.n_infill = n_infill
        self.eval_might_fail = problem.might_have_hidden_constraints()

        observer = ArchOptObserver(self.evaluate)
        search_space = self.get_search_space(problem)
        super().__init__(observer, search_space)

        self._results_folder = None
        self._datasets = None
        self._models = None
        self._state = None

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

    @property
    def search_space(self):
        return self._search_space

    @property
    def observer(self) -> 'MultiObserver':
        return self._observer

    @property
    def rule(self) -> 'AcquisitionRule':
        if self._rule is None:
            self._rule = self.get_acquisition_rule(pof=self.pof)
        return self._rule

    @rule.setter
    def rule(self, rule: 'AcquisitionRule'):
        self._rule = rule

    @property
    def is_constrained(self):
        return self._problem.n_ieq_constr > 0

    def initialize_from_previous(self, results_folder: str):
        capture_log()

        # Load from problem state
        population = self._problem.load_previous_results(results_folder)
        if population is not None:
            self._datasets = datasets = self._to_datasets(population)
            self._models = self.get_models(datasets)
            self._state = None
            log.info(f'Previous results loaded from problem results: {len(population)} design points')
            return

        # Load from optimizer state
        state_path = self.get_state_path(results_folder)
        if os.path.exists(state_path):
            try:
                results = FrozenRecord(pathlib.Path(state_path)).load()
            except UnpicklingError:
                log.exception(f'Could not load previous state from: {state_path}')
                return

            self._datasets = datasets = results.datasets
            self._models = results.models
            self._state = results.acquisition_state
            log.info(f'Previous results loaded from optimizer state: {self._get_n_points(datasets)} design points')
            return

        log.info('No previous results found')

    def run_optimization(self, results_folder=None) -> 'Record':
        """Runs a full optimization, including initial DOE"""
        if results_folder is not None:
            log.error('Currently not possible to pickle some TensorFlow models, disabling restart!')
            results_folder = None

        capture_log()
        self._results_folder = results_folder

        # Check how many points we already have available
        n_available = 0
        if self._datasets is not None:
            n_available = self._get_n_points(self._datasets)
            log.info(f'Starting optimization with {n_available} points already available')

        # Run (part of) DOE if needed
        if n_available < self.n_init:
            log.info(f'Running DOE: {self.n_init - n_available} points ({self.n_init} total)')
            datasets = self._run_doe(self.n_init - n_available)
            models = self.get_models(datasets)
            self._exec_callback(datasets, models)
        else:
            log.info(f'Skipping DOE, enough points available: {n_available} >= {self.n_init}')
            datasets = self._datasets
            models = self._models

        # Run (part of) optimization
        n_available = self._get_n_points(datasets)
        if n_available < self.n_init+self.n_infill:
            n_infill = self.n_infill - (n_available-self.n_init)
            log.info(f'Running optimization: {n_infill} infill points')
            opt_results = self.optimize(
                n_infill, datasets, models, self.rule, self._state,
                early_stop_callback=self._exec_callback, track_state=False)

            record = opt_results.final_result.unwrap()
            self._datasets = record.datasets
            self._models = record.models
            self._state = record.acquisition_state

        else:
            record = Record(datasets, models, acquisition_state=self._state)
            log.info(f'Skipping infill, enough points available: {n_available} >= {self.n_init}+{self.n_infill}')

        # Store final problem results
        if self._results_folder is not None:
            self._exec_callback(self._datasets, self._models, self._state)

        return record

    def _exec_callback(self, datasets, models, acquisition_state=None):
        self._datasets = datasets
        self._models = models
        self._state = acquisition_state

        # Store intermediate results if requested
        if self._results_folder is not None:
            # Store optimizer state
            Record(copy.deepcopy(datasets), copy.deepcopy(models), copy.deepcopy(acquisition_state))\
                .save(self.get_state_path(self._results_folder))

            # Store problem state
            self._problem.store_results(self._results_folder)

        return False

    def _run_doe(self, n: int):
        return self.observer(self.search_space.sample(n))

    def get_models(self, datasets):
        # https://secondmind-labs.github.io/trieste/1.0.0/notebooks/inequality_constraints.html#Modelling-the-two-functions
        search_space = self.search_space

        models = {}
        for tag, dataset in datasets.items():
            # https://secondmind-labs.github.io/trieste/1.0.0/notebooks/failure_ego.html#Build-GPflow-models
            if tag == FAILED:
                classifier = build_vgp_classifier(dataset, search_space, noise_free=True)
                models[tag] = VariationalGaussianProcess(
                    classifier, BatchOptimizer(tf.optimizers.Adam(1e-3)), use_natgrads=True)
                continue

            # https://secondmind-labs.github.io/trieste/1.0.0/notebooks/expected_improvement.html#Model-the-objective-function
            gpr = build_gpr(dataset, search_space, likelihood_variance=1e-7)
            models[tag] = GaussianProcessRegression(gpr, num_kernel_samples=100)

        return models

    @staticmethod
    def _get_n_points(datasets: Mapping[Hashable, 'Dataset']) -> int:
        if FAILED in datasets:
            return datasets[FAILED].query_points.shape[0]

        if OBJECTIVE not in datasets:
            return 0
        return datasets[OBJECTIVE].query_points.shape[0]

    @staticmethod
    def get_state_path(results_folder):
        return os.path.join(results_folder, 'trieste_state')

    def get_acquisition_rule(self, pof=.5) -> 'AcquisitionRule':
        """
        Builds the acquisition rule based on whether the problem is single- or multi-objective and constrained or not:
        https://secondmind-labs.github.io/trieste/1.0.0/notebooks/inequality_constraints.html#Define-the-acquisition-process
        https://secondmind-labs.github.io/trieste/1.0.0/notebooks/multi_objective_ehvi.html#Define-the-acquisition-function
        """

        if self._problem.n_eq_constr > 0:
            raise RuntimeError('Trieste currently does not support equality constraints')

        if self.is_constrained:
            # Reduce the PoF rules into one
            # https://secondmind-labs.github.io/trieste/1.0.0/notebooks/inequality_constraints.html#Constrained-optimization-with-more-than-one-constraint
            pof_builders = [ProbabilityOfFeasibility(threshold=pof).using(f'{CONSTR_PREFIX}{ig}')
                            for ig in range(self._problem.n_ieq_constr)]
            pof_builder = pof_builders[0] if len(pof_builders) == 1 else Product(*pof_builders)

            if self._problem.n_obj == 1:
                acq_builder = ExpectedConstrainedImprovement(OBJECTIVE, pof_builder)
            else:
                acq_builder = ExpectedConstrainedHypervolumeImprovement(OBJECTIVE, pof_builder)

        else:
            if self._problem.n_obj == 1:
                acq_builder = ExpectedImprovement().using(OBJECTIVE)
            else:
                acq_builder = ExpectedHypervolumeImprovement().using(OBJECTIVE)

        # Deal with hidden constraints in the acquisition function
        if self.eval_might_fail:
            pov = ProbabilityOfValidity().using(FAILED)
            acq_builder = Product(acq_builder, pov)

        return EfficientGlobalOptimization(acq_builder)

    @staticmethod
    def get_search_space(problem: ArchOptProblemBase) -> 'SearchSpace':
        box_buffer = []
        search_space: Optional['SearchSpace'] = None

        def _append_box():
            nonlocal box_buffer
            if len(box_buffer) == 0:
                return

            bounds = np.array(box_buffer)
            _append_space(Box(lower=tf.constant(bounds[:, 0], dtype=tf.float64),
                              upper=tf.constant(bounds[:, 1], dtype=tf.float64)))
            box_buffer = []

        def _append_space(space: 'SearchSpace'):
            nonlocal search_space
            if search_space is None:
                search_space = space
            else:
                search_space = search_space * space  # Creates a TaggedProductSearchSpace

        for i, var_def in enumerate(problem.des_vars):
            # We can have multiple real dimensions in one part of the design space, so we accumulate before actually
            # creating a Box (a continuous search space)
            if isinstance(var_def, var.Real):
                box_buffer.append(var_def.bounds)
                continue

            # Until there is a discrete dimension, which we add directly
            _append_box()
            if isinstance(var_def, var.Integer):
                points = np.arange(var_def.bounds[0], var_def.bounds[1]+1, dtype=int)

            elif isinstance(var_def, var.Binary):
                points = np.array([0, 1])

            elif isinstance(var_def, var.Choice):
                points = np.arange(0, len(var_def.options), dtype=int)

            else:
                raise RuntimeError(f'Unsupported design variable type: {var_def!r}')

            discrete_search_space = DiscreteSearchSpace(tf.constant(np.array([points]).T, dtype=tf.float64))
            _append_space(discrete_search_space)

        _append_box()

        if search_space is None:
            raise RuntimeError('Problem contains no design variables!')
        return search_space

    def evaluate(self, x: 'tf.Tensor') -> Dict[str, 'Dataset']:
        out = self._problem.evaluate(x.numpy(), return_as_dictionary=True)
        return self._process_evaluation_results(out)

    def _to_datasets(self, population: Population) -> Dict[str, 'Dataset']:
        return self._process_evaluation_results(population)

    def _process_evaluation_results(self, pop_or_dict: Union[dict, Population]) -> Dict[str, 'Dataset']:
        is_constrained = self.is_constrained

        # Separate failed evaluations (hidden constraints)
        is_failed = self._problem.get_failed_points(pop_or_dict)
        is_ok = ~is_failed
        x_all = pop_or_dict.get('X')
        x_out = x_all[is_ok, :]
        f = pop_or_dict.get('F')[is_ok, :]
        g = pop_or_dict.get('G')[is_ok, :] if is_constrained else None

        x_ts = tf.constant(x_out, dtype=tf.float64)
        datasets = {
            OBJECTIVE: Dataset(x_ts, tf.constant(f, dtype=tf.float64)),
            FAILED: Dataset(tf.constant(x_all, dtype=tf.float64), tf.cast(is_failed[:, None], dtype=tf.float64)),
        }

        if is_constrained:
            for ig in range(self._problem.n_ieq_constr):
                datasets[f'{CONSTR_PREFIX}{ig}'] = Dataset(x_ts, tf.constant(g[:, [ig]], dtype=tf.float64))

        return datasets

    def to_population(self, datasets: Dict[Hashable, 'Dataset']) -> Population:
        obj_dataset = datasets[OBJECTIVE]
        x = obj_dataset.query_points.numpy()
        kwargs = {
            'X': x,
            'F': obj_dataset.observations.numpy(),
        }

        if self.is_constrained:
            g = np.zeros((x.shape[0], self._problem.n_ieq_constr))
            for ig in range(self._problem.n_ieq_constr):
                g[:, ig] = datasets[f'{CONSTR_PREFIX}{ig}'].observations.numpy()

        return Population.new(**kwargs)


class ArchOptObserver:
    """
    The observer function that evaluates each architecture, according to the tagged observer pattern:
    https://secondmind-labs.github.io/trieste/1.0.0/notebooks/inequality_constraints.html

    Support for failed evaluations based on:
    https://secondmind-labs.github.io/trieste/1.0.0/notebooks/failure_ego.html#Define-the-data-sets

    Class needed to prevent overflow in BayesianOptimizer.__repr__
    """

    def __init__(self, func):
        self._func = func

    def __call__(self, x: 'tf.Tensor') -> Dict[str, 'Dataset']:
        return self._func(x)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ProbabilityOfValidity(SingleModelAcquisitionBuilder):
    """
    Acquisition function for dealing with failed regions (hidden constraints):
    https://secondmind-labs.github.io/trieste/1.0.0/notebooks/failure_ego.html#Create-a-custom-acquisition-function
    """

    def prepare_acquisition_function(self, model, dataset=None):

        def acquisition(at):
            mean, _ = model.predict_y(tf.squeeze(at, -2))
            return mean

        return acquisition
