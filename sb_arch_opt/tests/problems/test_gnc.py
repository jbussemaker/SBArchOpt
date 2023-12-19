import numpy as np
from sb_arch_opt.problems.gnc import *
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.sampling import HierarchicalSampling
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy


def test_gnc():
    problem: ArchOptProblemBase
    for problem, n_valid, imp_ratio in [
        (GNCNoActNrType(), 265, 1.93),
        (GNCNoActType(), 327, 14.1),
        (GNCNoActNr(), 26500, 14.1),
        (GNCNoAct(), 29857, 112.5),

        (GNCNoNrType(), 70225, 3.73),
        (GNCNoType(), 85779, 82.5),
        (GNCNoNr(), 70225000, 73.5),
        (GNC(), 79091323, 1761),

        (MDGNCNoActNr(), 265, 1.93),
        (MDGNCNoAct(), 327, 14.1),
        (SOMDGNCNoAct(), 327, 14.1),
        (MDGNCNoNr(), 70225, 3.73),
        (MDGNC(), 85779, 82.5),
    ]:
        run_test_hierarchy(problem, imp_ratio, check_n_valid=n_valid < 400)
        assert problem.get_n_valid_discrete() == n_valid

        assert np.any(~problem.is_conditionally_active)

        x_rnd, _ = problem.design_space.quick_sample_discrete_x(300)
        x_corr, is_act_corr = problem.correct_x(x_rnd)
        x_corr_, is_act_corr_ = problem.correct_x(x_corr)
        assert np.all(x_corr == x_corr_)
        assert np.all(is_act_corr == is_act_corr_)

        pop = HierarchicalSampling().do(problem, 300)
        if np.all(problem.is_discrete_mask):
            assert len(pop) == min(n_valid, 300)
        else:
            assert len(pop) == 300
