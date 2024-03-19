import os
import pytest
from sb_arch_opt.problems.rocket import *
from sb_arch_opt.problems.rocket_eval import *
from sb_arch_opt.tests.problems.test_hierarchical import run_test_hierarchy

def check_dependency():
    return pytest.mark.skipif(not HAS_ROCKET, reason='Rocket dependencies not installed')


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_slow_tests():
    assert HAS_ROCKET


@check_dependency()
def test_1_stage():
    rocket = Rocket(
        stages=[
            Stage(
                engines=[Engine.VULCAIN],
                length=26.47,
            ),
        ],
        head_shape=HeadShape.SPHERE,
        length_diameter_ratio=10.83,
        max_q=50e3,
        payload_density=2810.
    )

    performance = RocketEvaluator.evaluate(rocket)
    assert performance.cost == pytest.approx(53845830.)
    assert performance.payload_mass == pytest.approx(2783, abs=1)
    assert performance.delta_structural == pytest.approx(-23532, abs=1)
    assert performance.delta_payload == pytest.approx(-2.832, abs=1e-3)


@check_dependency()
def test_2_stages():
    rocket = Rocket(
        stages=[
            Stage(
                engines=[Engine.VULCAIN],
                length=20.37,
            ),
            Stage(
                engines=[Engine.VULCAIN],
                length=6.8,
            ),
        ],
        head_shape=HeadShape.SPHERE,
        length_diameter_ratio=11.32,
        max_q=50e3,
        payload_density=2810.
    )

    performance = RocketEvaluator.evaluate(rocket)
    assert performance.cost == pytest.approx(87563960.)
    assert performance.payload_mass == pytest.approx(7578, abs=1.)
    assert performance.delta_structural == pytest.approx(-27686, abs=1)
    assert performance.delta_payload == pytest.approx(-0.923, abs=1e-3)


@check_dependency()
def test_3_stages():
    rocket = Rocket(
        stages=[
            Stage(
                engines=[Engine.SRB],
                length=32.76,
            ),
            Stage(
                engines=[Engine.S_IVB],
                length=22.39,
            ),
            Stage(
                engines=[Engine.RS68],
                length=22.53,
            ),
        ],
        head_shape=HeadShape.ELLIPTICAL,
        ellipse_l_ratio=.175,
        length_diameter_ratio=17.55,
        max_q=50e3,
        payload_density=2810.
    )

    performance = RocketEvaluator.evaluate(rocket)
    assert performance.cost == pytest.approx(337894901.)
    assert performance.payload_mass == pytest.approx(57777, abs=1)
    assert performance.delta_structural == pytest.approx(-27031, abs=1)
    assert performance.delta_payload == pytest.approx(-118.9, abs=.1)


@check_dependency()
def test_rocket_problem():
    rocket = RocketArch()
    run_test_hierarchy(rocket, 2.83)


@check_dependency()
def test_lc_rocket_problem():
    rocket = LCRocketArch()
    run_test_hierarchy(rocket, 2.83)


@check_dependency()
def test_so_lc_rocket_problem():
    rocket = SOLCRocketArch(obj=RocketObj.OBJ_WEIGHTED)
    run_test_hierarchy(rocket, 2.83)
