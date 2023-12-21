import numpy as np
from sb_arch_opt.problems.gnc import *
from sb_arch_opt.sampling import HierarchicalSampling
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy


def test_gnc():
    problem: GNCProblemBase
    for problem, n_valid, imp_ratio, corr_ratio in [
        (GNCNoActNrType(), 265, 1.93, 1.93),
        (GNCNoActType(), 327, 14.1, 3.01),
        (GNCNoActNr(), 26500, 14.1, 14.1),
        (GNCNoAct(), 29857, 112.5, 17.2),

        (GNCNoNrType(), 70225, 3.73, 3.73),
        (GNCNoType(), 85779, 82.5, 9.01),
        (GNCNoNr(), 70225000, 73.5, 73.5),
        (GNC(), 79091323, 1761, 123),

        (MDGNCNoActNr(), 265, 1.93, 1.93),
        (MDGNCNoAct(), 327, 14.1, 3.01),
        (SOMDGNCNoAct(), 327, 14.1, 3.01),
        (MDGNCNoNr(), 70225, 3.73, 3.73),
        (MDGNC(), 85779, 82.5, 9.01),
    ]:
        run_test_hierarchy(problem, imp_ratio, check_n_valid=n_valid < 400, corr_ratio=corr_ratio)
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
            # Mixed-discrete problems
            assert len(pop) == 300
            assert problem.get_continuous_imputation_ratio() > 1
            assert problem.get_imputation_ratio() > imp_ratio
            assert problem.get_continuous_correction_ratio() > 1
            assert problem.get_correction_ratio() > corr_ratio

            if not problem.choose_type or not problem.choose_nr:
                assert problem.get_continuous_correction_ratio() == problem.get_continuous_imputation_ratio()
            else:
                assert problem.get_continuous_correction_ratio() != problem.get_continuous_imputation_ratio()


def test_gnc_cont_corr():
    problem = MDGNCNoActNr()
    assert np.all(problem.is_cont_mask[:6])

    x = np.array([problem.xl.copy()])
    x[0, :3] = [.5, .25, .75]
    x, is_active = problem.correct_x(x)
    assert np.all(x[0, :3] == [.5, .5, .75])
    assert np.all(is_active[0, :3])
