#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:35:37 2024

@author: psaves
"""

import logging
from typing import *
from sb_arch_opt.problem import *
from sb_arch_opt.sampling import *
from sb_arch_opt.algo.pymoo_interface import *
from sb_arch_opt.algo.pymoo_interface.metrics import EHVMultiObjectiveOutput

from pymoo.core.result import Result
from pymoo.core.sampling import Sampling
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.infill import InfillCriterion
from pymoo.core.termination import Termination
from pymoo.core.initialization import Initialization
from pymoo.core.duplicate import DuplicateElimination
from sb_arch_opt.algo.segomoe_interface import algo as algo_segomoe

from sb_arch_opt.algo.arch_sbo.infill import *
from sb_arch_opt.algo.arch_sbo.models import *

from sb_arch_opt.sampling import *
from sb_arch_opt.algo.arch_sbo.models import *

from sb_arch_opt.problem import ArchOptProblemBase

try:
    from smt.surrogate_models.surrogate_model import SurrogateModel
    from segomoe.sego import Sego
    from segomoe.constraint import Constraint
    from segomoe.sego_defs import get_sego_file_map, ExitStatus

    HAS_SEGOMOE = True
except ImportError:
    HAS_SEGOMOE = False


__all__ = [
    "HAS_SEGOMOE",
    "HAS_SMT",
    "check_dependencies",
    "SEGOMOEInterface",
    "InfillAlgorithm",
    "SBOInfill",
    "SurrogateInfillCallback",
    "SurrogateInfillOptimizationProblem",
]

log = logging.getLogger("sb_arch_opt.segomoe")


class InfillAlgorithm(Algorithm):
    """
    Algorithm that simpy uses some InfillCriterion to get infill points.
    The algorithm is compatible with the ask-tell interface.
    """

    def __init__(
        self,
        infill: InfillCriterion,
        problem: ArchOptProblemBase,
        results_folder: str,
        init_size: int,
        infill_size: int,
        use_moe=True,
        sego_options=None,
        model_options=None,
        verbose=True,
        init_sampling: Sampling = None,
        output=EHVMultiObjectiveOutput(),
        **kwargs,
    ):
        super(InfillAlgorithm, self).__init__(output=output, **kwargs)
        self.segomoe = algo_segomoe.SEGOMOEInterface(
            problem=problem,
            results_folder=results_folder,
            n_init=init_size,
            n_infill=infill_size,
        )

        self.init_size = init_size
        self.infill_size = infill_size or 1
        self.infill = infill
        if init_sampling is None:
            init_sampling = HierarchicalSampling()
        self.initialization = Initialization(
            init_sampling,
            repair=self.infill.repair,
            eliminate_duplicates=self.infill.eliminate_duplicates,
        )

        if self.output is None:
            from sb_arch_opt.algo.arch_sbo.metrics import SBOMultiObjectiveOutput

            self.output = SBOMultiObjectiveOutput()

    def initialize_from_previous_results(
        self, problem: ArchOptProblemBase, result_folder: str
    ):
        """Initialize the SBO algorithm from previously stored results"""
        return self.segomoe.initialize_from_previous(results_folder=results_folder)


class SBOInfill(InfillCriterion):
    """The main implementation of the SBO infill search"""

    def __init__(
        self,
        surrogate_model: "SurrogateModel",
        infill: SurrogateInfill,
        init_size: int,
        infill_size: int,
        results_folder: str,
        problem: ArchOptProblemBase,
        termination: Union[Termination, int] = None,
        eliminate_duplicates: DuplicateElimination = None,
        use_moe=True,
        sego_options=None,
        model_options=None,
        verbose=True,
        init_sampling: Sampling = None,
        output=EHVMultiObjectiveOutput(),
        **kwargs,
    ):
        if self.infill.eliminate_duplicates is None:
            eliminate_duplicates = LargeDuplicateElimination()
        super(SBOInfill, self).__init__(
            repair=self.infill.repair,
            eliminate_duplicates=eliminate_duplicates,
            **kwargs,
        )
        self.segomoe = algo_segomoe.SEGOMOEInterface(
            problem=problem,
            results_folder=results_folder,
            n_init=init_size,
            n_infill=infill_size,
        )

        self._is_init = None
        self.problem: Optional[ArchOptProblemBase] = None
        self.total_pop: Optional[Population] = None
        self._algorithm: Optional[Algorithm] = None

        self._surrogate_model_base = surrogate_model
        self._surrogate_model = None
        self.infill = infill

        self.x_train = None
        self.is_active_train = None
        self.y_train = None
        self.y_train_min = None
        self.y_train_max = None
        self.y_train_centered = None
        self.n_train = 0
        self.was_trained = False
        self.time_train = None
        self.time_infill = None
        self.pf_estimate = None

        self.pop_size = 100
        self.termination = termination
        self.verbose = verbose

        self.opt_results: Optional[List[Result]] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._exclude:
            state[key] = None
        return state

    def _generate_and_evaluate_infill_points(self, n_infill: int) -> Population:
        result = self.segomoe.run_optimization()
        if self.opt_results is None:
            self.opt_results = []
        self.opt_results.append(result)

        return opt_results

    def __repr__(self):
        return f"{self.__class__.__name__}, {self._problem!r})"
