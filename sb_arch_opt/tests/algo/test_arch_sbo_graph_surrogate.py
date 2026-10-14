import networkx as nx
import numpy as np
import pytest

pytest.importorskip("smt", reason="GraphKriging requires SMT.")

try:
    from smt.design_space import DesignSpace, IntegerVariable
except ImportError:
    from smt.utils.design_space import DesignSpace, IntegerVariable

from sb_arch_opt.algo.arch_sbo.graph import (
    CompositeGraphKernel,
    GraphKriging,
    GraphRepresentation,
    LdWloa,
    NodeLabeling,
    SizingFeature,
    SizingKernel,
)
from sb_arch_opt.algo.arch_sbo.models import MultiSurrogateModel


LABELING = NodeLabeling("kind", lambda node: node)


class ExampleDecoder:
    labelings = (LABELING,)
    sizing_features = (SizingFeature("size", "numeric"),)

    def decode(self, x):
        representations = []
        for row in x:
            graph = nx.Graph()
            graph.add_node(0)
            labels = {0: "root"}
            if row[0] >= 0.5:
                graph.add_edge(0, 1)
                labels[1] = "leaf"
            representations.append(
                GraphRepresentation(
                    graph,
                    {LABELING: labels},
                    np.asarray(row[:1]),
                )
            )
        return representations


def test_graph_kriging_defaults_to_cobyla():
    model = GraphKriging(ExampleDecoder(), LdWloa(0, [LABELING]))

    assert model.options["hyper_opt"] == "Cobyla"


def test_graph_kriging_train_and_predict():
    decoder = ExampleDecoder()
    kernel = CompositeGraphKernel(
        [
            LdWloa(1, [LABELING]),
            SizingKernel(decoder.sizing_features),
        ],
        composition="additive_interaction",
    )
    model = GraphKriging(
        decoder,
        kernel,
        hyper_opt="NoOp",
        poly="constant",
        corr="squar_exp",
        print_global=False,
        print_training=False,
        print_prediction=False,
        print_problem=False,
        print_solver=False,
        design_space=DesignSpace([IntegerVariable(0, 3)]),
    )
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([[0.0], [0.5], [0.5], [0.0]])

    model.set_training_values(x, y)
    model.train()
    prediction = model.predict_values(np.array([[0.0], [3.0]]))
    variance = model.predict_variances(np.array([[0.0], [3.0]]))

    assert prediction.shape == (2, 1)
    assert variance.shape == (2, 1)
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(variance))
    assert len(model.optimal_theta) == len(kernel.get_theta_parameters())
    np.testing.assert_array_equal(
        model.optimal_theta,
        [parameter.initial for parameter in kernel.get_theta_parameters()],
    )

    warm_start = np.linspace(0.2, 0.6, len(kernel.get_theta_parameters()))
    model.options["theta0"] = warm_start
    model.train()
    np.testing.assert_array_equal(model.optimal_theta, warm_start)


def test_multi_graph_kriging_reuses_wl_matrices_across_retraining():
    decoder = ExampleDecoder()
    model = GraphKriging(
        decoder,
        CompositeGraphKernel(
            [LdWloa(1, [LABELING]), SizingKernel(decoder.sizing_features)],
            composition="additive",
        ),
        hyper_opt="NoOp",
        poly="constant",
        print_global=False,
        print_training=False,
        print_prediction=False,
        print_problem=False,
        print_solver=False,
        design_space=DesignSpace([IntegerVariable(0, 3)]),
    )
    multi = MultiSurrogateModel(model)
    x = np.arange(4.0).reshape(-1, 1)
    y = np.column_stack((x[:, 0], x[:, 0] ** 2))

    multi.set_training_values(x[:3], y[:3])
    multi.train()
    old_first = multi._models[0].kernel.branches[0]._train_level_kernels
    assert multi._models[1].kernel.branches[0]._train_level_kernels is old_first

    multi.set_training_values(x, y)
    multi.train()
    new_first = multi._models[0].kernel.branches[0]._train_level_kernels
    assert multi._models[1].kernel.branches[0]._train_level_kernels is new_first
    for old_level, new_level in zip(old_first[0], new_first[0]):
        np.testing.assert_array_equal(new_level[:3, :3], old_level)
