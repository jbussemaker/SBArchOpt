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

This test suite contains a set of discrete, hierarchical, multi-objective test problems that represent various
architecting patterns, modeled as assignment (one or more source nodes to one or more target nodes) problems.
More information: https://github.com/jbussemaker/AssignmentEncoding

Test problems are based on:
- Common architecting patterns:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
- GN&C problem: see GNCProblemBase in gnc.py for an overview
"""
import numpy as np
from typing import *
from sb_arch_opt.problems.hierarchical import HierarchyProblemBase

try:
    from assign_pymoo.problem import AssignmentProblemBase, AssignmentProblem, MultiAssignmentProblem
    from assign_experiments.problems.gnc import *
    from assign_experiments.problems.analytical import *
    HAS_ASSIGN_ENC = True
except ImportError:
    HAS_ASSIGN_ENC = False

__all__ = ['HAS_ASSIGN_ENC', 'AssignmentProblemWrapper',
           'Assignment', 'AssignmentLarge', 'AssignmentInj', 'AssignmentInjLarge', 'AssignmentRepeat',
           'AssignmentRepeatLarge',
           'Partitioning', 'PartitioningLarge', 'PartitioningCovering', 'PartitioningCoveringLarge',
           'Downselecting', 'DownselectingLarge', 'Connecting', 'ConnectingUndirected', 'Permuting', 'PermutingLarge',
           'UnordNonReplComb', 'UnordNonReplCombLarge', 'UnorderedComb',
           'AssignmentGNCNoActType', 'AssignmentGNCNoAct', 'AssignmentGNCNoType', 'AssignmentGNC']


def check_dependency():
    if not HAS_ASSIGN_ENC:
        raise RuntimeError('assign_enc not installed: python setup.py install[assignment]')


class AssignmentProblemWrapper(HierarchyProblemBase):
    """Wrapper for the assignment problem definition in assign_enc"""
    _force_get_discrete_rates = False

    # Each assignment test problem has at least 1 constraint, which might be violated if a constraint-violation
    # imputation algorithm (i.e. this one doesn't actually impute, but rather sets the constraint to violated if an
    # invalid design vector is evaluated) is used. This is, however, more for testing purposes as this imputation
    # method is very ineffective for optimizers. Therefore, we can ignore this constraint.
    supress_invalid_matrix_constraint = True

    def __init__(self, problem: 'AssignmentProblemBase'):
        check_dependency()
        self._problem = problem

        n_con = problem.n_ieq_constr
        if self.supress_invalid_matrix_constraint:
            n_con -= 1
        super().__init__(problem.vars, n_obj=problem.n_obj, n_ieq_constr=n_con)

        self.design_space.use_auto_corrector = False

    def _get_n_valid_discrete(self) -> int:
        return self._problem.get_n_valid_design_points(n_cont=1)

    def _get_n_correct_discrete(self) -> int:
        pass

    def _is_conditionally_active(self) -> List[bool]:
        _, is_act_all = self.all_discrete_x
        if is_act_all is not None:
            return list(np.any(~is_act_all, axis=0))

        if isinstance(self._problem, MultiAssignmentProblem):
            dv_cond = []
            for assignment_manager in self._problem.assignment_managers:
                dv_cond += [dv.conditionally_active for dv in assignment_manager.design_vars]
            return dv_cond

        return [dv.conditionally_active for dv in self._problem.assignment_manager.design_vars]

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if isinstance(self._problem, AnalyticalProblemBase):
            x_all_map = self._problem.assignment_manager.get_all_design_vectors()
            if len(x_all_map) > 1:
                return

            x_all_map_main: np.ndarray = list(x_all_map.values())[0]
            is_active_all = x_all_map_main != -1
            x_all_map_main[~is_active_all] = 0

            return x_all_map_main, is_active_all

    def _print_extra_stats(self):
        if isinstance(self._problem, AssignmentProblem):
            # Distance correlation: correlation (pearson) between:
            # - Manhattan distances between design vectors, and;
            # - Manhattan distances between decoded assignment matrices
            # A higher correlation (100% is best) means that differences in design vectors better represent differences
            # in assignment matrices, which generally makes it easier for the optimizer to find an optimum
            dist_corr = self._problem.assignment_manager.encoder.get_distance_correlation()
            if dist_corr is not None:
                print(f'dist_corr    : {dist_corr*100:.0f}%')

        elif isinstance(self._problem, MultiAssignmentProblem):
            for i, assignment_manager in enumerate(self._problem.assignment_managers):
                dist_corr = assignment_manager.encoder.get_distance_correlation()
                if dist_corr is not None:
                    imp_ratio = assignment_manager.encoder.get_imputation_ratio()
                    print(f'dist_corr {i: 2d} : {dist_corr*100:.0f}%; '
                          f'imp_ratio : {imp_ratio:.2f}; {assignment_manager.encoder!s}')

        super()._print_extra_stats()

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        out = self._problem.evaluate(x, return_as_dictionary=True)
        x[:, :] = out['X']
        is_active_out[:, :] = out['is_active']
        f_out[:, :] = out['F']
        if self.n_ieq_constr > 0:
            g_eval = out['G']
            if self.supress_invalid_matrix_constraint:
                g_eval = g_eval[:, 1:]
            g_out[:, :] = g_eval

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        x[:, :], is_active[:, :] = self._problem.correct_x(x)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Assignment(AssignmentProblemWrapper):
    """
    Test problem representing the assignment pattern: assigning multiple sources to multiple targets, no restrictions on
    nr of connections per connector.

    Here, we assign 3 sources to 4 targets.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalAssignmentProblem(n_src=3, n_tgt=4))


class AssignmentLarge(AssignmentProblemWrapper):
    """
    Test problem representing the assignment pattern: assigning multiple sources to multiple targets, no restrictions on
    nr of connections per connector.

    Here, we assign 4 sources to 4 targets.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalAssignmentProblem(n_src=4, n_tgt=4))


class AssignmentInj(AssignmentProblemWrapper):
    """
    Test problem representing the injective assignment pattern: assigning multiple sources to multiple targets, each
    target receives max 1 assignment.

    Here, we assign 5 sources to 5 targets.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalAssignmentProblem(n_src=5, n_tgt=5, injective=True))


class AssignmentInjLarge(AssignmentProblemWrapper):
    """
    Test problem representing the injective assignment pattern: assigning multiple sources to multiple targets, each
    target receives max 1 assignment.

    Here, we assign 6 sources to 6 targets.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalAssignmentProblem(n_src=6, n_tgt=6, injective=True))


class AssignmentRepeat(AssignmentProblemWrapper):
    """
    Test problem representing the repeatable assignment pattern: assigning multiple sources to multiple targets, no
    restrictions on nr of connections per connector, connections can be repeated.

    Here, we assign 2 sources to 4 targets.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalAssignmentProblem(n_src=2, n_tgt=4, repeatable=True))


class AssignmentRepeatLarge(AssignmentProblemWrapper):
    """
    Test problem representing the repeatable assignment pattern: assigning multiple sources to multiple targets, no
    restrictions on nr of connections per connector, connections can be repeated.

    Here, we assign 2 sources to 5 targets.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalAssignmentProblem(n_src=2, n_tgt=5, repeatable=True))


class Partitioning(AssignmentProblemWrapper):
    """
    Test problem representing the partitioning pattern: divide a set of options into two or more non-covering sets,
    modeled by 2 or more sources (the sets) with any nr of connections, and targets (the options) with 1 connection
    each.

    Here, we partition 6 options (targets) in 4 sets (sources).

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalPartitioningProblem(n_src=4, n_tgt=6))


class PartitioningLarge(AssignmentProblemWrapper):
    """
    Test problem representing the partitioning pattern: divide a set of options into two or more non-covering sets,
    modeled by 2 or more sources (the sets) with any nr of connections, and targets (the options) with 1 connection
    each.

    Here, we partition 7 options (targets) in 5 sets (sources).

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalPartitioningProblem(n_src=5, n_tgt=7))


class PartitioningCovering(AssignmentProblemWrapper):
    """
    Test problem representing the covering partitioning pattern: divide a set of options into two or more sets that
    might overlap, modeled by 2 or more sources (the sets) with any nr of connections, and targets (the options) with 1
    or more connections each.

    Here, we partition 4 options (targets) in 3 sets (sources).

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalPartitioningProblem(n_src=3, n_tgt=4, covering=True))


class PartitioningCoveringLarge(AssignmentProblemWrapper):
    """
    Test problem representing the covering partitioning pattern: divide a set of options into two or more sets that
    might overlap, modeled by 2 or more sources (the sets) with any nr of connections, and targets (the options) with 1
    or more connections each.

    Here, we partition 4 options (targets) in 4 sets (sources).

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalPartitioningProblem(n_src=4, n_tgt=4, covering=True))


class Downselecting(AssignmentProblemWrapper):
    """
    Test problem representing the downselecting pattern: select zero or more elements from a set, modeled by one source
    with any nr of connections and targets with 0 or 1 connections each.

    Here, we downselect from 12 elements.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalDownselectingProblem(n_tgt=12))


class DownselectingLarge(AssignmentProblemWrapper):
    """
    Test problem representing the downselecting pattern: select zero or more elements from a set, modeled by one source
    with any nr of connections and targets with 0 or 1 connections each.

    Here, we downselect from 15 elements.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalDownselectingProblem(n_tgt=15))


class Connecting(AssignmentProblemWrapper):
    """
    Test problem representing the connecting pattern: connect among a set of elements, modeled by source and targets
    with any number of connections, however excluding connecting the same source (index) to target (index).

    Here, we connect among 4 elements (directed).

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalConnectingProblem(n=4))


class ConnectingUndirected(AssignmentProblemWrapper):
    """
    Test problem representing the undirected connecting pattern: connect among a set of elements, modeled by source and
    targets with any number of connections, however excluding connecting the same source (index) to target (index).
    Connections between elements are treated the same regardless of their direction.

    Here, we connect among 6 elements (undirected).

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalConnectingProblem(n=6, directed=False))


class Permuting(AssignmentProblemWrapper):
    """
    Test problem representing the permuting pattern: reorder a set of elements, modeled by the same amount of sources
    and targets, each having 1 connection.

    Here, we permute 7 elements.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalPermutingProblem(n=7))


class PermutingLarge(AssignmentProblemWrapper):
    """
    Test problem representing the permuting pattern: reorder a set of elements, modeled by the same amount of sources
    and targets, each having 1 connection.

    Here, we permute 8 elements.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalPermutingProblem(n=8))


class UnordNonReplComb(AssignmentProblemWrapper):
    """
    Test problem representing the unordered non-replacing combining pattern: take all combinations of a certain length
    from a set of elements that doesn't contain the same element twice, skipping permutations (reorderings) of already
    seen combinations, modeled by 1 source with n_take connections to targets with 0 or 1 connection each,
    no repetition.

    Here, we take 7 from 15.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalUnorderedNonReplaceCombiningProblem(n_take=7, n_tgt=15))


class UnordNonReplCombLarge(AssignmentProblemWrapper):
    """
    Test problem representing the unordered non-replacing combining pattern: take all combinations of a certain length
    from a set of elements that doesn't contain the same element twice, skipping permutations (reorderings) of already
    seen combinations, modeled by 1 source with n_take connections to targets with 0 or 1 connection each,
    no repetition.

    Here, we take 9 from 19.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalUnorderedNonReplaceCombiningProblem(n_take=9, n_tgt=19))


class UnorderedComb(AssignmentProblemWrapper):
    """
    Test problem representing the unordered combining pattern: take all combinations of a certain length
    from a set of elements, skipping permutations (reorderings) of already seen combinations, modeled by 1 source with
    n_take connections to targets with 0 or 1 connection each, repeated connections are allowed.

    Here, we take 5 from 10.

    More details about patterns in system architecting:
    Selva, Cameron & Crawley, "Patterns in System Architecture Decisions", 2017, DOI: 10.1002/sys.21370
    """

    def __init__(self):
        check_dependency()
        super().__init__(AnalyticalUnorderedCombiningProblem(n_take=5, n_tgt=10))


class AssignmentGNCNoActType(AssignmentProblemWrapper):
    """GN&C problem encoded using automatically-encoded optimal design variable encoders"""

    def __init__(self):
        check_dependency()
        super().__init__(GNCProblem(choose_type=False))


class AssignmentGNCNoAct(AssignmentProblemWrapper):
    """GN&C problem encoded using automatically-encoded optimal design variable encoders"""

    def __init__(self):
        check_dependency()
        super().__init__(GNCProblem())


class AssignmentGNCNoType(AssignmentProblemWrapper):
    """GN&C problem encoded using automatically-encoded optimal design variable encoders"""

    def __init__(self):
        check_dependency()
        super().__init__(GNCProblem(choose_type=False, actuators=True))


class AssignmentGNC(AssignmentProblemWrapper):
    """GN&C problem encoded using automatically-encoded optimal design variable encoders"""

    def __init__(self):
        check_dependency()
        super().__init__(GNCProblem(actuators=True))


if __name__ == '__main__':
    # # Assignment().print_stats()
    # # AssignmentLarge().print_stats()
    # # AssignmentInj().print_stats()
    # # AssignmentInjLarge().print_stats()
    # AssignmentRepeat().print_stats()
    # AssignmentRepeatLarge().print_stats()
    # # AssignmentLarge().plot_pf()

    # Partitioning().print_stats()
    # PartitioningLarge().print_stats()
    # PartitioningCovering().print_stats()
    # PartitioningCoveringLarge().print_stats()
    # Downselecting().print_stats()
    # DownselectingLarge().print_stats()
    # PartitioningCovering().plot_pf()

    # Connecting().print_stats()
    # ConnectingUndirected().print_stats()
    Permuting().print_stats()
    PermutingLarge().print_stats()

    # UnordNonReplComb().print_stats()
    # UnordNonReplCombLarge().print_stats()
    # UnorderedComb().print_stats()

    # AssignmentGNCNoActType().print_stats()
    # AssignmentGNCNoAct().print_stats()
    # AssignmentGNCNoType().print_stats()
    # AssignmentGNC().print_stats()
    # # AssignmentGNC().plot_pf()
