import numpy as np
import pytest

from sb_arch_opt.algo.arch_sbo.graph import HAS_NETWORKX, GraphDecoder, GraphRepresentation, NodeLabeling, SizingFeature

if HAS_NETWORKX:
    import networkx as nx


pytestmark = pytest.mark.skipif(not HAS_NETWORKX, reason="NetworkX dependency not installed")


def node_kind(node):
    return node["kind"]


def test_node_labeling():
    labeling = NodeLabeling("example.kind", node_kind)

    assert labeling({"kind": "component"}) == "component"
    assert labeling == NodeLabeling("example.kind", lambda _node: "other")


def test_graph_representation():
    labeling = NodeLabeling("example.kind", node_kind)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1])
    graph.add_edge(0, 1, kind="connects")
    graph.add_edge(0, 1, kind="connects")
    labels = {labeling: {0: "source", 1: "target"}}

    representation = GraphRepresentation(
        graph=graph,
        node_labels=labels,
        sizing_values=np.array([0.5, 1.0]),
    )

    assert representation.graph is graph
    assert representation.node_labels is labels
    assert representation.graph.number_of_edges(0, 1) == 2
    np.testing.assert_array_equal(representation.sizing_values, [0.5, 1.0])


def test_graph_representation_without_sizing():
    representation = GraphRepresentation(nx.Graph(), {})

    assert representation.sizing_values.shape == (0,)


def test_graph_decoder_protocol():
    labeling = NodeLabeling("example.kind", node_kind)

    class Decoder:
        labelings = (labeling,)
        sizing_features = (SizingFeature("size", "numeric"),)

        def decode(self, x):
            return [
                GraphRepresentation(
                    graph=nx.Graph(),
                    node_labels={labeling: {}},
                    sizing_values=np.asarray(row[:1], dtype=float),
                )
                for row in x
            ]

    decoder = Decoder()

    assert isinstance(decoder, GraphDecoder)
    assert len(decoder.decode(np.array([[0.25], [0.75]]))) == 2
