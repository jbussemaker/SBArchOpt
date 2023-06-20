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
from pymoo.core.indicator import Indicator
from pymoo.indicators.hv import Hypervolume
from pymoo.util.display.column import Column
from pymoo.core.termination import TerminateIfAny
from pymoo.termination.max_gen import MaximumGenerationTermination
from sb_arch_opt.algo.pymoo_interface.metrics import *

__all__ = ['EstimatedPFDistance', 'get_sbo_termination', 'PFDistanceTermination', 'SBOMultiObjectiveOutput']


def get_sbo_termination(n_max_infill: int, tol=1e-3, n_filter=2):
    return PFDistanceTermination(tol=tol, n_filter=n_filter, n_max_infill=n_max_infill)


class EstimatedPFDistance(Indicator):
    """Indicates the distance between the current Pareto front and the one estimated by the underlying model"""

    def __init__(self):
        super().__init__()
        self.algorithm = None

    def _do(self, f, *args, **kwargs):
        if self.algorithm is None:
            raise RuntimeError('Algorithm not set!')
        from sb_arch_opt.algo.arch_sbo.algo import InfillAlgorithm, SBOInfill

        if len(f) == 0:
            return 1

        if isinstance(self.algorithm, InfillAlgorithm) and isinstance(self.algorithm.infill_obj, SBOInfill):
            sbo_infill = self.algorithm.infill_obj
            pf_estimate = sbo_infill.get_pf_estimate()
            if pf_estimate is None:
                return 1

            hv = Hypervolume(pf=pf_estimate)
            hv_estimate = hv.do(pf_estimate)
            hv_f = hv.do(f)

            hv_dist = 1 - (hv_f / hv_estimate)
            if hv_dist < 0:
                hv_dist = 0
            return hv_dist

        return 0


class PFDistanceTermination(TerminateIfAny):
    """Termination criterion tracking the difference between the found and estimated Pareto fronts"""

    def __init__(self, tol=1e-3, n_filter=2, n_max_infill=100):
        self._pf_dist = EstimatedPFDistance()
        termination = [
            IndicatorDeltaToleranceTermination(SmoothedIndicator(self._pf_dist, n_filter=n_filter), tol),
            MaximumGenerationTermination(n_max_gen=n_max_infill),
        ]
        super().__init__(*termination)

    def update(self, algorithm):
        self._pf_dist.algorithm = algorithm
        return super().update(algorithm)


class SBOMultiObjectiveOutput(EHVMultiObjectiveOutput):
    """Extended multi-objective output for use with SBO"""

    def __init__(self):
        super().__init__()
        self.pf_dist_col = Column('pf_dist')
        self.pf_dist = EstimatedPFDistance()

    def initialize(self, algorithm):
        super().initialize(algorithm)
        self.pf_dist.algorithm = algorithm
        self.columns += [self.pf_dist_col]

    def update(self, algorithm):
        super().update(algorithm)

        f, feas = algorithm.opt.get("F", "feas")
        f = f[feas]

        self.pf_dist_col.set(self.pf_dist.do(f) if len(f) > 0 else None)
