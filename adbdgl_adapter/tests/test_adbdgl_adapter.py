from adbdgl_adapter.adbdgl_adapter import ArangoDB_DGL_Adapter
import pytest
from conftest import dgl, db, adbdgl_adapter

from dgl import DGLGraph


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, name, edge_definitions",
    [(adbdgl_adapter, "fraud-detection", None)],
)
def test_adb_graph_to_dgl(adapter: ArangoDB_DGL_Adapter, name: str, edge_definitions):
    assert_adapter_type(adapter)

    # Re-create the graph if defintions are provided
    if edge_definitions:
        db.delete_graph(name, ignore_missing=True)
        db.create_graph(name, edge_definitions=edge_definitions)

    arango_graph = db.graph(name)
    v_cols = arango_graph.vertex_collections()
    e_cols = {col["edge_collection"] for col in arango_graph.edge_definitions()}

    dgl_g: DGLGraph = adapter.arangodb_graph_to_dgl(name)
    assert_dgl_data(dgl_g, v_cols, e_cols)


def assert_adapter_type(adapter: ArangoDB_DGL_Adapter):
    assert type(adapter) is ArangoDB_DGL_Adapter


def assert_dgl_data(dgl_g: DGLGraph, v_cols, e_cols):
    for col in v_cols:
        assert dgl_g.num_nodes(col) == db.collection(col).count()

    for col in e_cols:
        assert dgl_g.num_edges(col) == db.collection(col).count()
