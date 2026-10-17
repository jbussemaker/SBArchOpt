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

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Dict, Hashable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .representation import GraphRepresentation, NodeLabeling, SizingFeature


__all__ = [
    "ThetaParameter",
    "GraphKernel",
    "LdWloa",
    "SizingKernel",
    "EdgeMultiplicityKernel",
    "CompositeGraphKernel",
]


class ThetaParameter(NamedTuple):
    """Bounds, optimizer scale, and initial value of one kernel parameter."""

    lower: float
    upper: float
    scale: str
    initial: float


class GraphKernel(ABC):
    """Compare decoded graphs for a graph surrogate.

    Values in ``theta`` follow the order from ``get_theta_parameters``.
    ``fit_transform`` sets the training graphs that later ``transform`` calls
    compare against.
    """

    def get_theta_parameters(self) -> Sequence[ThetaParameter]:
        """Describe tunable parameters in the order used by ``theta``."""
        return ()

    @abstractmethod
    def fit_transform(
        self,
        graphs: Sequence[GraphRepresentation],
        theta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit on training graphs and return their square kernel matrix.

        :param graphs: Training representations; rows and columns follow this order.
        :param theta: Kernel parameters, or ``None`` for implementation defaults.
        :return: Matrix with shape ``(n_train, n_train)``.
        """
        pass

    @abstractmethod
    def transform(
        self,
        graphs: Sequence[GraphRepresentation],
        theta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compare new graphs with the graphs from the last ``fit_transform``.

        :param graphs: New representations, in matrix-row order.
        :param theta: Kernel parameters, or ``None`` for implementation defaults.
        :return: Matrix with shape ``(n_new, n_train)``.
        """
        pass


@dataclass(frozen=True)
class _WLFeatures:
    counts: Tuple[Tuple[Counter, ...], ...]


@dataclass(frozen=True)
class _WLTrainingSnapshot:
    features: Tuple[_WLFeatures, ...]
    level_kernels: Tuple[Tuple[np.ndarray, ...], ...]


def _wl_features(structure_key, cutoff: int, intern) -> _WLFeatures:
    """Compute WL counts using compact labels shared across kernel copies."""
    labelings, neighbors = structure_key
    counts_by_labeling = []
    for labeling_index, raw_labels in enumerate(labelings):
        current = tuple(intern(("base", labeling_index, label)) for label in raw_labels)
        counts_by_level = []
        for level in range(cutoff + 1):
            counts_by_level.append(Counter(current))
            if level < cutoff:
                current = tuple(
                    intern((
                        "wl",
                        current[index],
                        tuple(sorted(current[neighbor] for neighbor in adjacent)),
                    ))
                    for index, adjacent in enumerate(neighbors)
                )
        counts_by_labeling.append(tuple(counts_by_level))
    return _WLFeatures(tuple(counts_by_labeling))


class _WLCache:
    """Shared by copies of one kernel; training data is never LRU-evicted."""

    def __init__(self, cache_size: Optional[int]):
        self.cache_size = cache_size
        self._label_ids = {}
        self.training_features = {}
        self.candidate_features = OrderedDict()
        self.training_snapshot: Optional[_WLTrainingSnapshot] = None

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def _intern(self, label) -> int:
        if label not in self._label_ids:
            self._label_ids[label] = len(self._label_ids)
        return self._label_ids[label]

    def features(self, structure_key, cutoff: int, training: bool) -> _WLFeatures:
        key = (cutoff, structure_key)
        features = self.training_features.get(key)
        if features is not None:
            return features

        features = self.candidate_features.get(key)
        if features is None:
            features = _wl_features(structure_key, cutoff, self._intern)
        if training:
            self.training_features[key] = features
            self.candidate_features.pop(key, None)
        else:
            self.candidate_features[key] = features
            self.candidate_features.move_to_end(key)
            if self.cache_size is not None and len(self.candidate_features) > self.cache_size:
                self.candidate_features.popitem(last=False)
        return features


def _same_objects(left, right):
    return (
        left is not None
        and right is not None
        and len(left) == len(right)
        and all(a is b for a, b in zip(left, right))
    )


class LdWloa(GraphKernel):
    """Learned-depth WLOA over one or more node labelings.

    Copies share theta-independent WL features and the latest training matrices.
    ``cache_size`` limits transient candidate graphs; ``None`` disables eviction.
    Training graph features are retained regardless of this limit.
    """

    def __init__(
        self,
        cutoff: int,
        labelings: Sequence[NodeLabeling],
        cache_size: Optional[int] = 8192,
    ):
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        if len(labelings) == 0:
            raise ValueError("at least one node labeling is required")
        if len({labeling.key for labeling in labelings}) != len(labelings):
            raise ValueError("node labeling keys must be unique")
        if cache_size is not None and cache_size < 0:
            raise ValueError("cache_size must be non-negative or None")

        self.cutoff = cutoff
        self.labelings = tuple(labelings)
        self._cache = _WLCache(cache_size)
        self._train_features: Optional[Tuple[_WLFeatures, ...]] = None
        self._train_level_kernels = None
        self._train_counts = None
        self._transform_features = None
        self._transform_level_kernels = None
        self._transform_counts = None

    @property
    def cache_size(self) -> Optional[int]:
        return self._cache.cache_size

    def get_theta_parameters(self) -> Sequence[ThetaParameter]:
        depth = [ThetaParameter(0.0, 1.0, "linear", 0.5)] * (self.cutoff + 1)
        mixture = [
            ThetaParameter(0.0, 1.0, "linear", (remaining - 1) / remaining)
            for remaining in range(len(self.labelings), 1, -1)
        ]
        return depth + mixture

    def fit_transform(self, graphs, theta=None):
        features = tuple(self._build_features(graph, training=True) for graph in graphs)
        counts = self._counts_by_labeling(features)
        self._train_level_kernels = self._train_kernels(features, counts)
        self._train_features = features
        self._train_counts = counts
        self._clear_transform_cache()
        return self._combine_kernels(
            self._train_level_kernels,
            counts,
            counts,
            theta,
        )

    def transform(self, graphs, theta=None):
        if self._train_features is None:
            raise RuntimeError("fit_transform must be called before transform")
        features = tuple(self._build_features(graph) for graph in graphs)
        if _same_objects(features, self._transform_features):
            counts = self._transform_counts
            level_kernels = self._transform_level_kernels
        else:
            counts = self._counts_by_labeling(features)
            level_kernels = tuple(
                tuple(
                    self._kernel_matrix_for_level(left, right)
                    for left, right in zip(left_by_level, right_by_level)
                )
                for left_by_level, right_by_level in zip(counts, self._train_counts)
            )
            self._transform_features = features
            self._transform_counts = counts
            self._transform_level_kernels = level_kernels
        return self._combine_kernels(
            level_kernels,
            counts,
            self._train_counts,
            theta,
        )

    def _build_features(
        self, representation: GraphRepresentation, training: bool = False
    ) -> _WLFeatures:
        graph = representation.graph
        nodes = tuple(graph.nodes)
        if graph.is_directed():
            neighbors = {
                node: tuple(set(graph.predecessors(node)) | set(graph.successors(node)))
                for node in nodes
            }
        else:
            neighbors = {node: tuple(graph.neighbors(node)) for node in nodes}

        node_index = {node: index for index, node in enumerate(nodes)}
        structure_key = (
            tuple(
                tuple(representation.node_labels[labeling][node] for node in nodes)
                for labeling in self.labelings
            ),
            tuple(
                tuple(sorted(node_index[neighbor] for neighbor in neighbors[node]))
                for node in nodes
            ),
        )
        return self._cache.features(structure_key, self.cutoff, training)

    def _combine_kernels(self, level_kernels, left_counts, right_counts, theta):
        depth_weights, labeling_weights = self._theta_parts(theta)
        matrices = [
            self._labeling_kernel(
                level_kernels[index],
                left_counts[index],
                right_counts[index],
                depth_weights,
            )
            for index in range(len(self.labelings))
        ]
        shape = level_kernels[0][0].shape
        result = np.zeros(shape, dtype=float)
        for weight, matrix in zip(labeling_weights, matrices):
            result += weight * matrix
        return result

    @staticmethod
    def _labeling_kernel(level_kernels, left_counts, right_counts, depth_weights):
        matrix = np.zeros_like(level_kernels[0], dtype=float)
        left_diag = np.zeros((len(left_counts[0]),), dtype=float)
        right_diag = np.zeros((len(right_counts[0]),), dtype=float)

        for weight, level_kernel, left, right in zip(
            depth_weights,
            level_kernels,
            left_counts,
            right_counts,
        ):
            matrix += weight * level_kernel
            left_diag += weight * np.fromiter(
                (sum(count.values()) for count in left),
                dtype=float,
                count=len(left),
            )
            right_diag += weight * np.fromiter(
                (sum(count.values()) for count in right),
                dtype=float,
                count=len(right),
            )

        denominator = np.sqrt(np.outer(left_diag, right_diag))
        return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0.0)

    def _counts_by_labeling(self, features):
        return tuple(
            tuple(
                tuple(feature.counts[labeling_index][level] for feature in features)
                for level in range(self.cutoff + 1)
            )
            for labeling_index in range(len(self.labelings))
        )

    def _train_kernels(self, features, counts):
        if _same_objects(features, self._train_features):
            return self._train_level_kernels

        snapshot = self._cache.training_snapshot
        old_features = snapshot.features if snapshot is not None else None
        old_kernels = snapshot.level_kernels if snapshot is not None else None
        if _same_objects(features, old_features):
            return old_kernels

        old_count = len(old_features) if old_features is not None else 0
        grows_previous_train = (
            old_count < len(features)
            and _same_objects(features[:old_count], old_features)
            and old_kernels is not None
        )
        if not grows_previous_train:
            kernels = tuple(
                tuple(self._kernel_matrix_for_level(level, level) for level in by_level)
                for by_level in counts
            )
        else:
            extended = []
            for labeling_index, by_level in enumerate(counts):
                labeling_kernels = []
                for level, level_counts in enumerate(by_level):
                    old_kernel = old_kernels[labeling_index][level]
                    old_counts = level_counts[:old_count]
                    new_counts = level_counts[old_count:]
                    new_old = self._kernel_matrix_for_level(new_counts, old_counts)
                    new_new = self._kernel_matrix_for_level(new_counts, new_counts)
                    kernel = np.empty((len(features), len(features)), dtype=float)
                    kernel[:old_count, :old_count] = old_kernel
                    kernel[old_count:, :old_count] = new_old
                    kernel[:old_count, old_count:] = new_old.T
                    kernel[old_count:, old_count:] = new_new
                    labeling_kernels.append(kernel)
                extended.append(tuple(labeling_kernels))
            kernels = tuple(extended)

        self._cache.training_snapshot = _WLTrainingSnapshot(features, kernels)
        return kernels

    def _clear_transform_cache(self):
        self._transform_features = None
        self._transform_level_kernels = None
        self._transform_counts = None

    @staticmethod
    def _kernel_matrix_for_level(left_counts, right_counts):
        labels = set()
        for counts in left_counts:
            labels.update(counts)
        for counts in right_counts:
            labels.update(counts)
        label_index = {label: index for index, label in enumerate(labels)}
        left = LdWloa._counts_to_csc(left_counts, label_index)
        right = LdWloa._counts_to_csc(right_counts, label_index)

        matrix = np.zeros((len(left_counts), len(right_counts)), dtype=float)
        for column in range(len(label_index)):
            left_start, left_stop = left.indptr[column], left.indptr[column + 1]
            right_start, right_stop = right.indptr[column], right.indptr[column + 1]
            if left_start == left_stop or right_start == right_stop:
                continue
            left_rows = left.indices[left_start:left_stop]
            right_rows = right.indices[right_start:right_stop]
            left_values = left.data[left_start:left_stop]
            right_values = right.data[right_start:right_stop]
            matrix[np.ix_(left_rows, right_rows)] += np.minimum(
                left_values[:, None],
                right_values[None, :],
            )
        return matrix

    @staticmethod
    def _counts_to_csc(counts, label_index):
        rows = []
        columns = []
        values = []
        for row, counter in enumerate(counts):
            for label, value in counter.items():
                rows.append(row)
                columns.append(label_index[label])
                values.append(float(value))
        return sparse.csc_matrix(
            (values, (rows, columns)),
            shape=(len(counts), len(label_index)),
        )

    def _theta_parts(self, theta):
        if theta is None:
            theta = [parameter.initial for parameter in self.get_theta_parameters()]
        theta = np.clip(np.asarray(theta, dtype=float).ravel(), 0.0, 1.0)
        n_depth = self.cutoff + 1
        if theta.size != n_depth + len(self.labelings) - 1:
            raise ValueError("incorrect number of WLOA parameters")

        remaining = 1.0
        labeling_weights = []
        for fraction in theta[n_depth:]:
            labeling_weights.append(remaining * (1.0 - fraction))
            remaining *= fraction
        labeling_weights.append(remaining)
        return theta[:n_depth], np.asarray(labeling_weights)


class SizingKernel(GraphKernel):
    """Kernel over normalized numeric and categorical sizing features."""

    def __init__(
        self,
        features: Sequence[SizingFeature],
        power: float = 2.0,
    ):
        if len(features) == 0:
            raise ValueError("at least one sizing feature is required")
        if not 0.0 < power <= 2.0:
            raise ValueError("sizing power must be in (0, 2]")
        self.features = tuple(features)
        self.power = power
        self._numeric = np.array([feature.kind == "numeric" for feature in features])
        self._categorical = ~self._numeric
        self._train_graphs = None
        self._train_values: Optional[np.ndarray] = None
        self._train_powered_distances = None
        self._transform_graphs = None
        self._transform_powered_distances = None

    def get_theta_parameters(self) -> Sequence[ThetaParameter]:
        return [ThetaParameter(1e-6, 20.0, "log", 1.0)] * len(self.features)

    def fit_transform(self, graphs, theta=None):
        graphs = tuple(graphs)
        if not _same_objects(graphs, self._train_graphs):
            values = self._arrays(graphs)
            self._train_powered_distances = self._training_distances(graphs, values)
            self._train_graphs = graphs
            self._train_values = values
            self._clear_transform_cache()
        return self._kernel_from_distances(self._train_powered_distances, theta)

    def transform(self, graphs, theta=None):
        if self._train_values is None:
            raise RuntimeError("fit_transform must be called before transform")
        graphs = tuple(graphs)
        if not _same_objects(graphs, self._transform_graphs):
            values = self._arrays(graphs)
            self._transform_powered_distances = self._powered_distances(
                values[:, None, :],
                self._train_values[None, :, :],
            )
            self._transform_graphs = graphs
        return self._kernel_from_distances(self._transform_powered_distances, theta)

    def _arrays(self, graphs):
        if len(graphs) == 0:
            shape = (0, len(self.features))
            return np.empty(shape)
        return np.asarray([graph.sizing_values for graph in graphs], dtype=float)

    def _training_distances(self, graphs, values):
        old_graphs = self._train_graphs
        old_count = len(old_graphs) if old_graphs is not None else 0
        grows_previous_train = (
            old_count < len(graphs)
            and _same_objects(graphs[:old_count], old_graphs)
            and self._train_powered_distances is not None
        )
        if not grows_previous_train:
            return self._powered_distances(values[:, None, :], values[None, :, :])

        distances = np.empty((len(graphs), len(graphs), len(self.features)), dtype=float)
        distances[:old_count, :old_count] = self._train_powered_distances
        new_old = self._powered_distances(
            values[old_count:, None, :],
            values[None, :old_count, :],
        )
        distances[old_count:, :old_count] = new_old
        distances[:old_count, old_count:] = np.swapaxes(new_old, 0, 1)
        distances[old_count:, old_count:] = self._powered_distances(
            values[old_count:, None, :],
            values[None, old_count:, :],
        )
        return distances

    def _powered_distances(self, left_values, right_values):
        distances = np.zeros(np.broadcast_shapes(left_values.shape, right_values.shape), dtype=float)
        delta = np.abs(left_values - right_values)
        distances[..., self._numeric] = delta[..., self._numeric]
        distances[..., self._categorical] = (
            left_values[..., self._categorical] != right_values[..., self._categorical]
        )

        return distances**self.power

    def _kernel_from_distances(self, powered_distances, theta):
        if theta is None:
            theta = [parameter.initial for parameter in self.get_theta_parameters()]
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != len(self.features):
            raise ValueError(f"SizingKernel theta size must be {len(self.features)}, got {theta.size}")
        return np.exp(-np.tensordot(powered_distances, theta, axes=([-1], [0])))

    def _clear_transform_cache(self):
        self._transform_graphs = None
        self._transform_powered_distances = None


class EdgeMultiplicityKernel(GraphKernel):
    """Kernel over multiplicities of a selected edge kind."""

    def __init__(self, edge_kind: Hashable, endpoint_labeling: NodeLabeling, gamma0: float = 1.0):
        self.edge_kind = edge_kind
        self.endpoint_labeling = endpoint_labeling
        self.gamma0 = gamma0
        self._train_features: Optional[Tuple[Dict[Tuple[Hashable, Hashable], float], ...]] = None
        self._feature_cache = {}
        self._train_distances = None
        self._transform_features = None
        self._transform_distances = None

    def get_theta_parameters(self) -> Sequence[ThetaParameter]:
        return [ThetaParameter(1e-6, 20.0, "log", self.gamma0)]

    def fit_transform(self, graphs, theta=None):
        features = tuple(self._features(graph) for graph in graphs)
        if not _same_objects(features, self._train_features):
            self._train_distances = self._training_distances(features)
            self._train_features = features
            self._clear_transform_cache()
        return self._kernel(self._train_distances, theta)

    def transform(self, graphs, theta=None):
        if self._train_features is None:
            raise RuntimeError("fit_transform must be called before transform")
        features = tuple(self._features(graph) for graph in graphs)
        if not _same_objects(features, self._transform_features):
            self._transform_distances = self._distances(features, self._train_features)
            self._transform_features = features
        return self._kernel(self._transform_distances, theta)

    def _features(self, representation):
        cached = self._feature_cache.get(id(representation))
        if cached is not None and cached[0] is representation:
            return cached[1]

        graph = representation.graph
        counts = defaultdict(int)
        edges = graph.edges(keys=True, data=True) if graph.is_multigraph() else graph.edges(data=True)
        for edge in edges:
            source, target, data = (edge[0], edge[1], edge[-1])
            if data.get("kind") == self.edge_kind:
                counts[(source, target)] += 1

        labels = representation.node_labels[self.endpoint_labeling]
        values = defaultdict(float)
        for (source, target), multiplicity in counts.items():
            values[(labels[source], labels[target])] += math.log1p(multiplicity)
        features = dict(values)
        self._feature_cache[id(representation)] = (representation, features)
        return features

    def _training_distances(self, features):
        old_features = self._train_features
        old_count = len(old_features) if old_features is not None else 0
        grows_previous_train = (
            old_count < len(features)
            and _same_objects(features[:old_count], old_features)
            and self._train_distances is not None
        )
        if not grows_previous_train:
            return self._distances(features, features)

        distances = np.empty((len(features), len(features)), dtype=float)
        distances[:old_count, :old_count] = self._train_distances
        new_old = self._distances(features[old_count:], features[:old_count])
        distances[old_count:, :old_count] = new_old
        distances[:old_count, old_count:] = new_old.T
        distances[old_count:, old_count:] = self._distances(
            features[old_count:],
            features[old_count:],
        )
        return distances

    @staticmethod
    def _distances(left, right):
        distances = np.empty((len(left), len(right)), dtype=float)
        for i, values_left in enumerate(left):
            for j, values_right in enumerate(right):
                keys = set(values_left) | set(values_right)
                distances[i, j] = sum(
                    abs(values_left.get(key, 0.0) - values_right.get(key, 0.0))
                    for key in keys
                )
        return distances

    def _kernel(self, distances, theta):
        if theta is None:
            gamma = self.gamma0
        else:
            theta = np.asarray(theta, dtype=float).ravel()
            if theta.size != 1:
                raise ValueError(f"EdgeMultiplicityKernel theta size must be 1, got {theta.size}")
            gamma = float(theta[0])
        return np.exp(-gamma * distances)

    def _clear_transform_cache(self):
        self._transform_features = None
        self._transform_distances = None


class CompositeGraphKernel(GraphKernel):
    """Composition of structural, sizing, and edge graph-kernel branches."""

    def __init__(self, branches: Sequence[GraphKernel], composition: str = "additive"):
        if len(branches) == 0:
            raise ValueError("at least one kernel branch is required")
        if composition not in {"additive", "additive_interaction", "multiplicative"}:
            raise ValueError(f"unsupported graph kernel composition: {composition!r}")
        self.branches = tuple(branches)
        self.composition = composition

    def get_theta_parameters(self) -> Sequence[ThetaParameter]:
        parameters: List[ThetaParameter] = []
        if self._uses_weights():
            parameters.extend(
                [ThetaParameter(0.01, 100.0, "linear", 1.0)] * (self._term_count() - 1)
            )
        for branch in self.branches:
            parameters.extend(branch.get_theta_parameters())
        return parameters

    def fit_transform(self, graphs, theta=None):
        weights, branch_theta = self._theta_parts(theta)
        matrices = [
            branch.fit_transform(graphs, values)
            for branch, values in zip(self.branches, branch_theta)
        ]
        return self._compose(matrices, weights)

    def transform(self, graphs, theta=None):
        weights, branch_theta = self._theta_parts(theta)
        matrices = [
            branch.transform(graphs, values)
            for branch, values in zip(self.branches, branch_theta)
        ]
        return self._compose(matrices, weights)

    def _uses_weights(self):
        return self.composition in {"additive", "additive_interaction"} and self._term_count() > 1

    def _term_count(self):
        if self.composition == "additive_interaction":
            return len(self.branches) + len(tuple(combinations(self.branches, 2)))
        return len(self.branches)

    def _theta_parts(self, theta):
        parameters = self.get_theta_parameters()
        if theta is None:
            theta = [parameter.initial for parameter in parameters]
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != len(parameters):
            raise ValueError(
                f"CompositeGraphKernel theta size must be {len(parameters)}, got {theta.size}"
            )

        cursor = 0
        weights = None
        if self._uses_weights():
            n_free = self._term_count() - 1
            free_weights = np.maximum(theta[:n_free], 0.0)
            raw_weights = np.concatenate(([1.0], free_weights))
            weights = raw_weights / np.sum(raw_weights)
            cursor = n_free

        branch_theta = []
        for branch in self.branches:
            size = len(branch.get_theta_parameters())
            branch_theta.append(theta[cursor:cursor + size])
            cursor += size
        return weights, branch_theta

    def _compose(self, matrices, weights):
        if self.composition == "multiplicative":
            return np.prod(matrices, axis=0)

        terms = list(matrices)
        if self.composition == "additive_interaction":
            terms.extend(left * right for left, right in combinations(matrices, 2))
        if weights is None:
            return terms[0]
        return sum(weight * term for weight, term in zip(weights, terms))
