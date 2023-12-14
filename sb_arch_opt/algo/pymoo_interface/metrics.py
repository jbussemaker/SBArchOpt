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
from pymoo.core.problem import Problem
from pymoo.core.indicator import Indicator
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from pymoo.util.display.column import Column
from pymoo.core.termination import TerminateIfAny
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.normalization import ZeroToOneNormalization
from pymoo.termination.delta import DeltaToleranceTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.default import DefaultSingleObjectiveTermination

from sb_arch_opt.problem import ArchOptProblemBase

__all__ = ['get_default_termination', 'SmoothedIndicator', 'IndicatorDeltaToleranceTermination', 'EstimateHV',
           'DeltaHVTermination', 'EHVMultiObjectiveOutput']


def get_default_termination(problem: Problem, xtol=5e-4, cvtol=1e-8, tol=1e-4, n_iter_check=5, n_max_gen=100,
                            n_max_eval: int = None):
    if problem.n_obj == 1:
        return DefaultSingleObjectiveTermination(
            xtol=xtol, cvtol=cvtol, ftol=tol, period=n_iter_check, n_max_gen=n_max_gen, n_max_evals=n_max_eval)

    return DeltaHVTermination(tol=tol, n_max_gen=n_max_gen, n_max_eval=n_max_eval)
    # return DefaultMultiObjectiveTermination(
    #     xtol=xtol, cvtol=cvtol, ftol=ftol, period=n_iter_check, n_max_gen=n_max_gen, n_max_evals=n_max_eval)


class SmoothedIndicator(Indicator):
    """Smooths an underlying indicator using an exponential moving average"""

    def __init__(self, indicator: Indicator, n_filter: int = 5):
        self.indicator = indicator
        self.n_filter = n_filter
        super().__init__()
        self.data = []

    @property
    def alpha(self):
        return 1/(1+self.n_filter)

    def _do(self, f, *args, **kwargs):
        value = self.indicator.do(f, *args, **kwargs)
        if value is None or np.isnan(value):
            return np.nan
        self.data.append(value)

        weight_factor = 1-self.alpha
        next_weight = 1

        ema = self.data[0]
        weight = 1.
        for value in self.data[1:]:
            weight *= weight_factor
            if ema != value:
                ema = ((weight * ema) + (next_weight * value)) / (weight + next_weight)
            weight += next_weight

        return ema


class IndicatorDeltaToleranceTermination(DeltaToleranceTermination):
    """Delta tolerance termination based on some indicator"""

    def __init__(self, indicator: Indicator, tol, n_skip=0):
        self.indicator = indicator
        super().__init__(tol, n_skip=n_skip)
        self.scale = None

    def _delta(self, prev, current):
        delta = np.abs(prev - current)
        if np.isnan(delta):
            return 100

        # Scale the value to improve percentage calculation
        if self.scale is None:
            # At the current value, delta-tol should be 100 (--> 1%)
            self.scale = (100-self.tol)/(delta-self.tol)

        delta = (delta-self.tol)*self.scale + self.tol
        return delta

    def _data(self, algorithm):
        f, feas = algorithm.opt.get("F", "feas")
        return self.indicator.do(f[feas])


class EstimateHV(Hypervolume):
    """An indicator for the Hypervolume without knowing the initial reference point"""

    def __init__(self):
        super().__init__(ref_point=1, norm_ref_point=False)
        self.ref_point = None

    def do(self, f, *args, **kwargs):
        if f.ndim == 1:
            f = f[None, :]

        if self.ref_point is None:
            f_invalid = np.any(np.isinf(f) | np.isnan(f), axis=1)
            f_valid = f[~f_invalid, :]
            if len(f_valid) == 0:
                return np.nan

            nadir = np.max(f[~f_invalid, :], axis=0)
            ideal = np.min(f[~f_invalid, :], axis=0)

            self.normalization = ZeroToOneNormalization(ideal, nadir)
            self.ref_point = self.normalization.forward(nadir)

        return super().do(f, *args, **kwargs)


class DeltaHVTermination(TerminateIfAny):
    """
    Termination criterion tracking the difference in HV improvement, filtered by an EMA. For more information, see:

    J.H. Bussemaker et al., "Effectiveness of Surrogate-Based Optimization Algorithms for System Architecture
    Optimization", AIAA Aviation 2021, DOI: [10.2514/6.2021-3095](https://arc.aiaa.org/doi/10.2514/6.2021-3095)
    """

    def __init__(self, tol=1e-4, n_filter=2, n_max_gen=100, n_max_eval: int = None):
        termination = [
            IndicatorDeltaToleranceTermination(SmoothedIndicator(EstimateHV(), n_filter=n_filter), tol),
            MaximumGenerationTermination(n_max_gen=n_max_gen),
        ]
        if n_max_eval is not None:
            termination += [
                MaximumFunctionCallTermination(n_max_evals=n_max_eval),
            ]
        super().__init__(*termination)


class EHVMultiObjectiveOutput(MultiObjectiveOutput):
    """Multi-objective output that also displays the estimated HV and some population statistics"""

    def __init__(self, pop_stats=True):
        super().__init__()
        self.ehv_col = Column('hv_est')
        self.estimate_hv = EstimateHV()

        self.pop_stat_cols = []
        if pop_stats:
            self.pop_stat_cols = [Column('not_failed'), Column('feasible'), Column('optimal')]

    def initialize(self, algorithm):
        super().initialize(algorithm)
        self.columns += [self.ehv_col]+self.pop_stat_cols

    def update(self, algorithm):
        super().update(algorithm)

        f, feas = algorithm.opt.get("F", "feas")
        f = f[feas]

        self.ehv_col.set(self.estimate_hv.do(f) if len(f) > 0 else None)

        if len(self.pop_stat_cols) > 0:
            pop_stats = ArchOptProblemBase.get_population_statistics(
                algorithm.pop if algorithm.pop is not None else Population.new())
            stats = [f'{row[0]} ({row[1]})' for row in pop_stats.iloc[1:, 1:3].values]
            for i, stat in enumerate(stats):
                self.pop_stat_cols[i].set(stat)
