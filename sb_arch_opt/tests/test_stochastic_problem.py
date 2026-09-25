import pytest
import numpy as np

try:
    import openturns as ot
    from sb_arch_opt.uncertainty import *
    from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
    from sb_arch_opt.tests.conftest import (VectorizedProblem, HierarchicalProblem,
                                            DeterministicResponseProblem, make_space)

except ImportError:
    pytest.skip(
        "SBArchOpt uncertainty package is not installed! Run: pip install sb-arch-opt[uncertainty]",
        allow_module_level=True,
    )


def _output(values):
    return StochasticOutput.from_output_samples(ot.Sample(np.array(values, dtype=float).reshape((-1, 1))))


def test_parameter_space():
    space = StochasticParameterSpace([
        StochasticParameter('a', ot.Normal(0., 1.)),
        StochasticParameter('b', ot.Uniform(0., 1.)),
    ])

    assert space.n_parameters == 2
    assert space.parameter_names == ['a', 'b']
    assert space.joint_dist.getDimension() == 2
    assert [parameter.name for parameter in space.parameters] == ['a', 'b']

    for samples in [space.get_random_samples(20), space.get_lhs_samples(20)]:
        assert samples.shape == (20, 2)
        assert np.all(np.isfinite(samples))


def test_scalars():
    out = _output([1., 2., 3., 4., 5.])
    mean, std = out.mean, out.std

    assert Mean().scalarize(out) == pytest.approx(mean)
    assert Quantile(q=.9).scalarize(out) == pytest.approx(out.quantile(.9))
    assert Margin(k=2.).scalarize(out) == pytest.approx(out.margin(2.))

    # The margin is the conservative value, so which tail that is depends on the direction
    assert Margin(k=2.).scalarize(out) == pytest.approx(mean + 2. * std)
    assert Margin(k=2., direction=-1).scalarize(out) == pytest.approx(mean + 2. * std)
    assert Margin(k=2., direction=1).scalarize(out) == pytest.approx(mean - 2. * std)

    # Each instance carries its own parameters
    assert Quantile(q=.5).scalarize(out) != Quantile(q=.95).scalarize(out)
    assert Margin(k=1.).scalarize(out) != Margin(k=3.).scalarize(out)

    with pytest.raises(ValueError):
        Quantile(q=2.)


def test_scalars_penalize_spread():
    wide, narrow = _output([1., 3., 5.]), _output([2.5, 3., 3.5])
    assert wide.mean == pytest.approx(narrow.mean)

    assert Margin(k=2.).scalarize(narrow) < Margin(k=2.).scalarize(wide)  # minimized: lower is better
    assert Margin(k=2., direction=1).scalarize(narrow) > Margin(k=2., direction=1).scalarize(wide)  # maximized


@pytest.mark.parametrize('method_name', ['MonteCarlo', 'PolynomialChaos'])
def test_uq_method_samples(method_name):
    method_class = {'MonteCarlo': MonteCarlo, 'PolynomialChaos': PolynomialChaos}[method_name]
    space = make_space(ot.Normal(0., 1.), ot.Normal(0., 1.))
    method = method_class(n_evaluations=40, seed=42)

    samples = method.get_samples(space)
    assert samples.shape == (40, 2)

    # Drawn once and reused, so every design point sees the same realizations (common random numbers)
    assert np.all(method.get_samples(space) == samples)

    method.resample()
    assert method.get_samples(space).shape == (40, 2)

    with pytest.raises(ValueError):
        method.get_samples(None)


def test_parameter_realization():
    space = StochasticParameterSpace([
        StochasticParameter('a', ot.Normal(0., 1.)),
        StochasticParameter('b', ot.Uniform(2., 4.)),
    ])
    samples = MonteCarlo(n_evaluations=5, seed=42).get_samples(space)

    for i in range(5):
        parameters = space.param_realization(samples, i)

        assert [parameter.name for parameter in parameters] == ['a', 'b']
        # Column j of the design belongs to parameter j
        assert [parameter.sample for parameter in parameters] == pytest.approx(list(samples[i, :]))
        assert all(isinstance(parameter.sample, float) for parameter in parameters)


def test_process_results():
    space = make_space(ot.Normal(0., 1.))
    method = MonteCarlo(n_evaluations=3, seed=42)

    outputs = method.process_results(np.array([[1., 10., 20.], [2., 11., 21.], [3., 12., 22.]]), space)

    assert len(outputs) == 3  # one output per response column
    assert [output.mean for output in outputs] == pytest.approx([2., 11., 21.])
    assert all(output.method_results is None for output in outputs)  # Monte Carlo has nothing beyond the samples

def test_user_only_implements_arch_evaluate_sample(stochastic_problem):
    out = stochastic_problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert out['F'].shape == (2, 1)
    assert np.all(np.isfinite(out['F']))
    # E[(u-x0)^2] + x1^2 = (E[u]-x0)^2 + var + x1^2
    assert out['F'][0, 0] == pytest.approx(.05**2, abs=2e-3)
    assert out['F'][1, 0] == pytest.approx(1. + .05**2, abs=2e-2)


def test_multiple_design_points_at_once(stochastic_problem):
    x = np.random.RandomState(0).uniform(-2., 2., size=(7, 2))
    out = stochastic_problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (7, 1)
    assert np.all(np.isfinite(out['F']))
    assert len(np.unique(out['F'])) == 7


def test_evaluation_is_deterministic_and_seeded():
    x = np.array([[.5, .5]])
    f_a = VectorizedProblem(n=50, seed=3).evaluate(x, return_as_dictionary=True)['F']
    f_b = VectorizedProblem(n=50, seed=3).evaluate(x, return_as_dictionary=True)['F']
    f_c = VectorizedProblem(n=50, seed=4).evaluate(x, return_as_dictionary=True)['F']

    assert np.allclose(f_a, f_b)
    assert not np.allclose(f_a, f_c)


def test_resample_draws_new_samples():
    problem = VectorizedProblem(n=50, seed=3)
    x = np.array([[.5, .5]])
    f_a = problem.evaluate(x, return_as_dictionary=True)['F'].copy()

    problem.uq_method.resample()
    problem.uq_method.seed = None
    assert not np.allclose(f_a, problem.evaluate(x, return_as_dictionary=True)['F'])


def test_hierarchical_problem_with_constraint(hierarchical_problem):
    x = np.array([[0, .5, .3], [1, .9, .35]])
    out = hierarchical_problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (2, 1)
    assert out['G'].shape == (2, 1)
    assert np.all(np.isfinite(out['F']))
    assert np.all(np.isfinite(out['G']))


def test_response_kinds_use_their_own_scalar(all_response_kinds_problem):
    problem = all_response_kinds_problem
    out = problem.evaluate(np.array([[.5]]), return_as_dictionary=True)

    # Each response kind is published in its own object array, laid out like F, G and H
    assert out['F'][0, 0] == pytest.approx(Mean().scalarize(out['f_stochastic'][0, 0]))
    assert out['G'][0, 0] == pytest.approx(Mean().scalarize(out['g_stochastic'][0, 0]))
    assert out['H'][0, 0] == pytest.approx(Margin(k=3.).scalarize(out['h_stochastic'][0, 0]))

    # The equality constraint uses a margin, so it is above its own mean
    assert out['H'][0, 0] > out['h_stochastic'][0, 0].mean


def test_scalar_counts_checked_per_response_kind():
    with pytest.raises(ValueError):
        HierarchicalProblem(ieq_constr_scalar=[Mean(), Mean()])
    with pytest.raises(ValueError):
        HierarchicalProblem(obj_scalar=[Mean(), Mean()])
    with pytest.raises(ValueError):
        HierarchicalProblem(ieq_constr_scalar=['quantile'])

    HierarchicalProblem(ieq_constr_scalar=[Mean()])  # a matching count is accepted


def test_statistics_available_per_design_point(stochastic_problem):
    out = stochastic_problem.evaluate(np.array([[1., 0.], [0., 0.]]), return_as_dictionary=True)

    assert out['f_stochastic'].shape == (2, 1)
    assert all(isinstance(output, StochasticOutput) for output in out['f_stochastic'].ravel())
    assert out['g_stochastic'].shape == (2, 0)

    # The realizations really reach the model, so the response scatters
    for output in out['f_stochastic'][:, 0]:
        assert output.std > 0.


def test_reported_statistics_reproduce_the_reduced_value():
    problem = VectorizedProblem(n=200, obj_scalar=[Margin(k=2.)])
    out = problem.evaluate(np.array([[1., 0.], [.5, .5]]), return_as_dictionary=True)

    for i, output in enumerate(out['f_stochastic'][:, 0]):
        assert out['F'][i, 0] == pytest.approx(Margin(k=2.).scalarize(output))
        assert out['F'][i, 0] == pytest.approx(output.mean + 2.*output.std)


def test_stochastic_output_statistics():
    out = _output(np.linspace(0., 10., 101))

    assert out.mean == pytest.approx(5.)
    # The kernel widens the fitted distribution slightly compared to the samples it was built from
    assert out.std == pytest.approx(np.std(np.linspace(0., 10., 101)), rel=.1)
    assert out.quantile(.5) == pytest.approx(5., abs=.1)
    assert out.dist.computeCDF(5.) == pytest.approx(.5, abs=.05)
    assert str(out) == f'(mean = {out.mean:.4g}, sigma = {out.std:.4g})'


@pytest.mark.parametrize('method_name', ['MonteCarlo', 'PolynomialChaos'])
def test_a_response_that_does_not_depend_on_the_parameters(method_name):
    uq_method = {'MonteCarlo': MonteCarlo, 'PolynomialChaos': PolynomialChaos}[method_name](50, seed=3)
    # A response that is constant over the realizations has no distribution to fit: it is reported as the value
    # itself, so no scalarization can add a margin to it
    problem = DeterministicResponseProblem(uq_method=uq_method, obj_scalar=[Mean(), Margin(k=2.)])
    out = problem.evaluate(np.array([[.5], [1.]]), return_as_dictionary=True)

    assert np.all(np.isfinite(out['F']))
    assert not problem.get_failed_points(out).any()

    for i, x_i in enumerate([.5, 1.]):
        varying, constant = out['f_stochastic'][i, :]
        assert isinstance(varying, StochasticOutput)
        assert varying.std > 0.

        assert isinstance(constant, float)
        assert constant == pytest.approx(2.*x_i)
        assert out['F'][i, 1] == pytest.approx(2.*x_i)


def test_a_failed_sample_fails_the_design_point():
    # Any non-finite sample discards the design point: with a single failed evaluation the model is not
    # trustworthy at that point, and SBArchOpt treats NaN as a hidden-constraint violation
    problem = VectorizedProblem(n=50, fail=True)
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    # There is no distribution to fit, so the response comes back as NaN rather than as an output
    assert not np.isfinite(out['f_stochastic'][0, 0])
    assert not np.isfinite(out['F'][0, 0])
    assert problem.get_failed_points(out)[0]


def test_pce_is_more_accurate_than_monte_carlo():
    # Both methods draw the same kind of LHS design, so this compares fitting an expansion against averaging
    # over it. f = (u-x0)^2 + x1^2 is quadratic in u, so the expansion is essentially exact.
    x = np.array([[1., 0.]])
    exact = .05**2  # E[(u-x)^2] at x = E[u] is Var[u]

    f_mc = VectorizedProblem(n=20, obj_scalar=[Mean()]).evaluate(x, return_as_dictionary=True)['F'][0, 0]
    f_pce = VectorizedProblem(uq_method=PolynomialChaos(20, seed=42), obj_scalar=[Mean()]) \
        .evaluate(x, return_as_dictionary=True)['F'][0, 0]

    assert abs(f_pce - exact) < abs(f_mc - exact)
    assert f_pce == pytest.approx(exact, abs=1e-4)


def test_pce_through_the_problem():
    # Regression: the problem called process_results without the parameter space, which PCE requires
    problem = VectorizedProblem(uq_method=PolynomialChaos(50, seed=42), obj_scalar=[Mean()])
    x = np.array([[.5, .5]])
    out = problem.evaluate(x, return_as_dictionary=True)

    assert np.all(np.isfinite(out['F']))
    assert np.allclose(out['F'], problem.evaluate(x, return_as_dictionary=True)['F'])

    # Statistics come from the fitted expansion, not from the 50 expensive evaluations
    assert isinstance(out['f_stochastic'][0, 0].method_results, ot.FunctionalChaosResult)


def test_pce_terms_and_sample_requirement():
    space = make_space(*[ot.Normal(0., 1.)]*3)

    assert PolynomialChaos(200, degree=8).n_terms(space) == 165  # degree 8 in 3 dimensions
    assert PolynomialChaos(200, degree=8).n_terms(make_space(ot.Normal(0., 1.))) == 9  # ... in 1 dimension

    with pytest.raises(ValueError):
        PolynomialChaos(10, degree=8).get_samples(space)  # 165 terms cannot be fitted from 10 samples


def test_pce_provides_chaos_result_for_sensitivity_analysis():
    problem = VectorizedProblem(uq_method=PolynomialChaos(50, seed=42), obj_scalar=[Mean()])
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    chaos_result = out['f_stochastic'][0, 0].method_results
    assert isinstance(chaos_result, ot.FunctionalChaosResult)

    sobol = ot.FunctionalChaosSobolIndices(chaos_result)
    assert sobol.getSobolIndex(0) == pytest.approx(1., abs=1e-6)  # a single parameter explains everything


def test_pce_fails_the_design_point_when_evaluations_fail():
    # A response with failed evaluations cannot be fitted; the design point then fails as usual
    problem = VectorizedProblem(uq_method=PolynomialChaos(50, seed=42), obj_scalar=[Mean()], fail=True)
    out = problem.evaluate(np.array([[.5, .5]]), return_as_dictionary=True)

    assert not np.isfinite(out['f_stochastic'][0, 0])
    assert not np.isfinite(out['F'][0, 0])