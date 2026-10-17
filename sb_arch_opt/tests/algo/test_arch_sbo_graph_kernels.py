import copy

import numpy as np
import pytest

from sb_arch_opt.algo.arch_sbo.graph import (
    HAS_NETWORKX,
    CompositeGraphKernel,
    EdgeMultiplicityKernel,
    GraphRepresentation,
    LdWloa,
    NodeLabeling,
    SizingFeature,
    SizingKernel,
)

if HAS_NETWORKX:
    import networkx as nx


pytestmark = pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX dependency not installed")


TYPE = NodeLabeling("type", lambda node: node)
FAMILY = NodeLabeling("family", lambda node: node)


def make_graph(edges, type_labels, family_labels=None, sizing=()):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(type_labels)
    for source, target, kind in edges:
        graph.add_edge(source, target, kind=kind)
    labels = {TYPE: type_labels}
    if family_labels is not None:
        labels[FAMILY] = family_labels
    return GraphRepresentation(
        graph=graph,
        node_labels=labels,
        sizing_values=np.asarray(sizing, dtype=float),
    )


def test_ld_wloa_single_labeling():
    graphs = [
        make_graph([(0, 1, "derives")], {0: "a", 1: "b"}),
        make_graph([(0, 1, "derives")], {0: "a", 1: "b"}),
        make_graph([], {0: "a", 1: "b"}),
    ]
    kernel = LdWloa(cutoff=1, labelings=[TYPE])

    matrix = kernel.fit_transform(graphs)

    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] < 1.0
    assert np.min(np.linalg.eigvalsh(matrix)) > -1e-10
    np.testing.assert_allclose(kernel.transform(graphs), matrix)
    assert len(kernel.get_theta_parameters()) == 2


def test_ld_wloa_multiple_labelings_share_depth_parameters():
    graphs = [
        make_graph([], {0: "same"}, {0: "family-a"}),
        make_graph([], {0: "same"}, {0: "family-b"}),
    ]
    kernel = LdWloa(cutoff=2, labelings=[TYPE, FAMILY])

    matrix = kernel.fit_transform(graphs)

    assert matrix[0, 1] < 1.0
    assert len(kernel.get_theta_parameters()) == 4  # 3 depth + 1 labeling-mixture


def test_ld_wloa_rejects_duplicate_labeling_keys():
    with pytest.raises(ValueError, match="labeling keys must be unique"):
        LdWloa(1, [TYPE, NodeLabeling("type", lambda node: str(node))])


def test_ld_wloa_reuses_level_kernels(monkeypatch):
    graphs = [
        make_graph([(0, 1, "derives")], {0: "a", 1: "b"}),
        make_graph([], {0: "a", 1: "b"}),
        make_graph([(1, 0, "derives")], {0: "a", 1: "c"}),
    ]
    kernel = LdWloa(cutoff=1, labelings=[TYPE])
    calls = []
    kernel_matrix_for_level = kernel._kernel_matrix_for_level

    def track(left, right):
        calls.append((len(left), len(right)))
        return kernel_matrix_for_level(left, right)

    monkeypatch.setattr(kernel, "_kernel_matrix_for_level", track)
    kernel.fit_transform(graphs[:2], theta=np.array([0.2, 0.8]))
    first_call_count = len(calls)

    kernel.fit_transform(graphs[:2], theta=np.array([0.8, 0.2]))
    assert len(calls) == first_call_count

    incremental = kernel.fit_transform(graphs, theta=np.array([0.8, 0.2]))
    assert calls[first_call_count:] == [(1, 2), (1, 1)] * 2

    expected = LdWloa(cutoff=1, labelings=[TYPE]).fit_transform(
        graphs,
        theta=np.array([0.8, 0.2]),
    )
    np.testing.assert_allclose(incremental, expected)


def test_ld_wloa_reuses_transform_kernels(monkeypatch):
    train = [make_graph([], {0: "a"}), make_graph([], {0: "b"})]
    test = [make_graph([], {0: "c"})]
    kernel = LdWloa(cutoff=1, labelings=[TYPE])
    kernel.fit_transform(train)
    calls = []
    kernel_matrix_for_level = kernel._kernel_matrix_for_level

    def track(left, right):
        calls.append((len(left), len(right)))
        return kernel_matrix_for_level(left, right)

    monkeypatch.setattr(kernel, "_kernel_matrix_for_level", track)
    kernel.transform(test, theta=np.array([0.2, 0.8]))
    first_call_count = len(calls)
    kernel.transform(test, theta=np.array([0.8, 0.2]))

    assert first_call_count == 2
    assert len(calls) == first_call_count


def test_ld_wloa_copies_share_graph_feature_cache():
    graph = make_graph([(0, 1, "derives")], {0: "a", 1: "b"})
    kernel = LdWloa(cutoff=1, labelings=[TYPE])
    copied = copy.deepcopy(kernel)

    original_features = kernel._build_features(graph)
    copied_features = copied._build_features(graph)

    assert copied_features is original_features


def test_ld_wloa_kernel_copies_share_graph_features():
    first_graph = make_graph([(0, 1, "derives")], {0: "shared-a", 1: "shared-b"})
    same_graph = make_graph([(0, 1, "derives")], {0: "shared-a", 1: "shared-b"})
    template = LdWloa(cutoff=2, labelings=[TYPE])
    first = copy.deepcopy(template)
    second = copy.deepcopy(template)

    assert first._build_features(first_graph) is second._build_features(same_graph)


def test_ld_wloa_kernel_copies_share_compact_labels_after_eviction():
    graph = make_graph([(0, 1, "derives")], {0: "a", 1: "b"})
    template = LdWloa(cutoff=2, labelings=[TYPE], cache_size=0)
    first = copy.deepcopy(template)
    second = copy.deepcopy(template)

    first_features = first._build_features(graph)
    second_features = second._build_features(graph)

    assert first_features is not second_features
    assert first_features.counts == second_features.counts
    assert all(
        isinstance(label, int)
        for level in first_features.counts[0]
        for label in level
    )


def test_ld_wloa_kernel_copies_reuse_and_extend_training_matrices(monkeypatch):
    graphs = [
        make_graph([], {0: "matrix-a"}),
        make_graph([], {0: "matrix-b"}),
        make_graph([], {0: "matrix-c"}),
    ]
    template = LdWloa(cutoff=3, labelings=[TYPE], cache_size=1)
    first = copy.deepcopy(template)
    first.fit_transform(graphs[:2])

    # Candidate searches may overflow the LRU without evicting training data.
    first._build_features(make_graph([], {0: "candidate-a"}))
    first._build_features(make_graph([], {0: "candidate-b"}))
    assert len(template._cache.candidate_features) == 1

    second = copy.deepcopy(template)
    calls = []
    matrix_for_level = second._kernel_matrix_for_level

    def track(left, right):
        calls.append((len(left), len(right)))
        return matrix_for_level(left, right)

    monkeypatch.setattr(second, "_kernel_matrix_for_level", track)
    second.fit_transform(graphs[:2])
    assert calls == []

    incremental = second.fit_transform(graphs)
    assert calls == [(1, 2), (1, 1)] * 4
    np.testing.assert_allclose(incremental, incremental.T)
    np.testing.assert_allclose(np.diag(incremental), 1.0)


def test_ld_wloa_cache_size_validation():
    with pytest.raises(ValueError, match="cache_size must be non-negative"):
        LdWloa(cutoff=1, labelings=[TYPE], cache_size=-1)

    unlimited = LdWloa(cutoff=0, labelings=[TYPE], cache_size=None)
    unlimited._build_features(make_graph([], {0: "unlimited-a"}))
    unlimited._build_features(make_graph([], {0: "unlimited-b"}))
    assert unlimited.cache_size is None
    assert len(unlimited._cache.candidate_features) == 2


def test_sizing_kernel():
    features = [SizingFeature("size", "numeric"), SizingFeature("material", "categorical")]
    graphs = [
        make_graph([], {}, sizing=[0.0, 0]),
        make_graph([], {}, sizing=[1.0, 1]),
    ]
    kernel = SizingKernel(features)

    matrix = kernel.fit_transform(graphs, theta=np.ones(2))

    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix[0, 1], np.exp(-2.0))
    np.testing.assert_allclose(kernel.transform(graphs, theta=np.ones(2)), matrix)


def test_sizing_kernel_rejects_invalid_parameters():
    graph = make_graph([], {}, sizing=[0.5])
    with pytest.raises(ValueError, match="power must be"):
        SizingKernel([SizingFeature("size", "numeric")], power=2.5)
    kernel = SizingKernel([SizingFeature("size", "numeric")])
    with pytest.raises(ValueError, match="theta size must be 1, got 2"):
        kernel.fit_transform([graph], theta=np.array([1.0, 2.0]))


def test_sizing_kernel_reuses_and_extends_distances():
    features = [SizingFeature("size", "numeric")]
    graphs = [
        make_graph([], {}, sizing=[0.0]),
        make_graph([], {}, sizing=[0.5]),
        make_graph([], {}, sizing=[1.0]),
    ]
    kernel = SizingKernel(features, power=1.9)
    kernel.fit_transform(graphs[:2], theta=np.array([0.5]))
    cached_distances = kernel._train_powered_distances

    kernel.fit_transform(graphs[:2], theta=np.array([2.0]))
    assert kernel._train_powered_distances is cached_distances

    incremental = kernel.fit_transform(graphs, theta=np.array([2.0]))
    expected = SizingKernel(features, power=1.9).fit_transform(
        graphs,
        theta=np.array([2.0]),
    )
    np.testing.assert_allclose(incremental, expected)


def test_edge_multiplicity_kernel():
    graphs = [
        make_graph([(0, 1, "connects")], {0: "provided", 1: "needed"}),
        make_graph(
            [(0, 1, "connects"), (0, 1, "connects")],
            {0: "provided", 1: "needed"},
        ),
    ]
    kernel = EdgeMultiplicityKernel("connects", TYPE)

    matrix = kernel.fit_transform(graphs, theta=np.ones(1))

    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix[0, 1], np.exp(-abs(np.log(2.0) - np.log(3.0))))
    np.testing.assert_allclose(kernel.transform(graphs, theta=np.ones(1)), matrix)


def test_edge_multiplicity_kernel_rejects_extra_parameter():
    graph = make_graph([], {0: "provided"})
    with pytest.raises(ValueError, match="theta size must be 1, got 2"):
        EdgeMultiplicityKernel("connects", TYPE).fit_transform(
            [graph], theta=np.array([1.0, 2.0])
        )


def test_edge_multiplicity_kernel_reuses_and_extends_distances():
    graphs = [
        make_graph([(0, 1, "connects")], {0: "provided", 1: "needed"}),
        make_graph(
            [(0, 1, "connects"), (0, 1, "connects")],
            {0: "provided", 1: "needed"},
        ),
        make_graph([], {0: "provided", 1: "needed"}),
    ]
    kernel = EdgeMultiplicityKernel("connects", TYPE)
    kernel.fit_transform(graphs[:2], theta=np.array([0.5]))
    cached_distances = kernel._train_distances

    kernel.fit_transform(graphs[:2], theta=np.array([2.0]))
    assert kernel._train_distances is cached_distances

    incremental = kernel.fit_transform(graphs, theta=np.array([2.0]))
    expected = EdgeMultiplicityKernel("connects", TYPE).fit_transform(
        graphs,
        theta=np.array([2.0]),
    )
    np.testing.assert_allclose(incremental, expected)


def test_composite_graph_kernel():
    features = [SizingFeature("size", "numeric")]
    graphs = [
        make_graph([], {0: "a"}, sizing=[0.0]),
        make_graph([], {0: "b"}, sizing=[1.0]),
    ]

    structure_expected = LdWloa(0, [TYPE]).fit_transform(graphs, theta=np.array([0.5]))
    sizing_expected = SizingKernel(features).fit_transform(graphs, theta=np.array([1.0]))
    kernel = CompositeGraphKernel(
        [LdWloa(0, [TYPE]), SizingKernel(features)],
        composition="additive",
    )

    matrix = kernel.fit_transform(graphs, theta=np.array([1.0, 0.5, 1.0]))

    np.testing.assert_allclose(matrix, 0.5 * structure_expected + 0.5 * sizing_expected)
    np.testing.assert_allclose(kernel.transform(graphs, theta=np.array([1.0, 0.5, 1.0])), matrix)


def test_composite_graph_kernel_rejects_invalid_configuration():
    graph = make_graph([], {0: "a"})
    with pytest.raises(ValueError, match="unsupported graph kernel composition"):
        CompositeGraphKernel([LdWloa(0, [TYPE])], composition="unknown")

    kernel = CompositeGraphKernel([LdWloa(0, [TYPE])])
    with pytest.raises(ValueError, match="theta size must be 1, got 2"):
        kernel.fit_transform([graph], theta=np.array([0.5, 0.5]))


def test_composite_graph_kernel_clamps_negative_weights():
    features = [SizingFeature("size", "numeric")]
    graphs = [
        make_graph([], {0: "a"}, sizing=[0.0]),
        make_graph([], {0: "b"}, sizing=[1.0]),
    ]
    structure_expected = LdWloa(0, [TYPE]).fit_transform(
        graphs,
        theta=np.array([0.5]),
    )
    kernel = CompositeGraphKernel(
        [LdWloa(0, [TYPE]), SizingKernel(features)],
        composition="additive",
    )

    matrix = kernel.fit_transform(graphs, theta=np.array([-1.0, 0.5, 1.0]))

    np.testing.assert_allclose(matrix, structure_expected)
    assert np.min(np.linalg.eigvalsh(matrix)) > -1e-10
