"""
MIT License

Copyright: (c) 2026, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


__all__ = ["HAS_NETWORKX", "NodeLabeling", "SizingFeature", "GraphRepresentation", "GraphDecoder"]


@dataclass(frozen=True)
class NodeLabeling:
    """A named rule for labeling nodes of a domain graph.

    :param key: Identifier used to match labels from a decoder with a kernel.
        Use a different key when the meaning of the labels changes.
    :param function: Maps a source-graph node to a hashable label. A decoder
        can call it when building each graph representation.
    """

    key: str
    function: Callable[[Any], Hashable] = field(compare=False, hash=False, repr=False)

    def __call__(self, node: Any) -> Hashable:
        return self.function(node)


@dataclass(frozen=True)
class SizingFeature:
    """Description of one sizing coordinate exported by a graph decoder.

    :param name: Human-readable name of the coordinate.
    :param kind: ``numeric`` compares the difference between values;
        ``categorical`` only checks whether values match.
    """

    name: str
    kind: Literal["numeric", "categorical"]


def _empty_float_array() -> np.ndarray:
    return np.empty((0,), dtype=float)


@dataclass(frozen=True)
class GraphRepresentation:
    """Kernel inputs for one decoded design, independent of its source domain.

    :param graph: NetworkX graph whose nodes and edges describe the design.
    :param node_labels: For each labeling declared by the decoder, a mapping
        from every node in ``graph`` to its hashable label.
    :param sizing_values: One-dimensional array with one value per decoder
        sizing feature, in the order of ``GraphDecoder.sizing_features``.

    Do not change the graph, labels, or sizing values after decoding. Kernels
    may reuse values computed from them.
    """

    graph: "nx.Graph"
    node_labels: Mapping[NodeLabeling, Mapping[Hashable, Hashable]]
    sizing_values: np.ndarray = field(default_factory=_empty_float_array)


@runtime_checkable
class GraphDecoder(Protocol):
    """Domain adapter that converts design vectors into graph-kernel inputs.

    Implement this protocol for a new design space. Its labelings and sizing
    features describe the data present in every representation it returns.
    """

    @property
    def labelings(self) -> Sequence[NodeLabeling]:
        """Labeling rules available in every decoded representation."""
        ...

    @property
    def sizing_features(self) -> Sequence[SizingFeature]:
        """Sizing coordinates, in the order used by ``sizing_values``."""
        ...

    def decode(self, x: np.ndarray) -> Sequence[GraphRepresentation]:
        """Return one representation per row of ``x``, in the same order.

        :param x: Design vectors with shape ``(n_designs, n_variables)``.
        """
        ...
