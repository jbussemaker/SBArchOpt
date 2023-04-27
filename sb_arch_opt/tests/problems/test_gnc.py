import numpy as np
from sb_arch_opt.problems.gnc import *
from sb_arch_opt.problem import ArchOptProblemBase
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy


def test_gnc():
    problem: ArchOptProblemBase
    for problem, n_valid, imp_ratio in [
        (GNCNoActNrType(), 265, 989),
        (GNCNoActType(), 327, 7.2e3),
        (GNCNoActNr(), 26500, 7.2e3),
        (GNCNoAct(), 29857, 57.6e3),

        (GNCNoNrType(), 70225, 1911),
        (GNCNoType(), 85779, 42.2e3),
        (GNCNoNr(), 70225000, 37.6e3),
        (GNC(), 79091323, 901e3),
    ]:
        run_test_hierarchy(problem, imp_ratio, check_n_valid=False)
        assert problem.get_n_valid_discrete() == n_valid

        assert np.any(~problem.is_conditionally_active)
