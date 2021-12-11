import pytest
from conftest import ArangoDB_DGL_Adapter, db, conn, adbdgl_adapter

from dgl import DGLGraph


@pytest.mark.unit
def test_validate_attributes():
    bad_connection = {
        "dbName": "_system",
        "hostname": "localhost",
        "protocol": "http",
        "port": 8529,
        # "username": "root",
        # "password": "password",
    }

    with pytest.raises(ValueError):
        ArangoDB_DGL_Adapter(bad_connection)


@pytest.mark.unit
def test_validate_controller_class():
    class Bad_ADBDGL_Controller:
        pass

    with pytest.raises(TypeError):
        ArangoDB_DGL_Adapter(conn, Bad_ADBDGL_Controller)


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, name, attributes",
    [
        (
            adbdgl_adapter,
            "fraud-detection",
            {
                "vertexCollections": {
                    "account": {"rank"},
                    "Class": {},
                    "customer": {"Sex", "Ssn", "rank"},
                },
                "edgeCollections": {
                    "accountHolder": {},
                    "Relationship": {
                        "label",
                        "name",
                        "relationshipType",
                    },
                    "transaction": {},
                },
            },
        ),
    ],
)
def test_adb_to_dgl(adapter: ArangoDB_DGL_Adapter, name: str, attributes: dict):
    assert_adapter_type(adapter)
    dgl_g = adapter.arangodb_to_dgl(name, attributes)

    for col, atribs in attributes["vertexCollections"].items():
        num_nodes = dgl_g.num_nodes(col)

        assert num_nodes == db.collection(col).count()
        for atrib in atribs:
            assert atrib in dgl_g.ndata
            assert col in dgl_g.ndata[atrib]
            assert len(dgl_g.ndata[atrib][col]) == num_nodes

    for col, atribs in attributes["edgeCollections"].items():
        num_edges = dgl_g.num_edges(col)

        assert num_edges == db.collection(col).count()
        for atrib in atribs:
            assert atrib in dgl_g.edata
            tup_key = [tup for tup in dgl_g.canonical_etypes if col in tup][0]
            assert tup_key in dgl_g.edata[atrib]
            assert len(dgl_g.edata[atrib][tup_key]) == num_edges


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, name, vcols, ecols",
    [
        (
            adbdgl_adapter,
            "fraud-detection",
            {"account", "Class", "customer"},
            {"accountHolder", "Relationship", "transaction"},
        )
    ],
)
def test_adb_collections_to_dgl(
    adapter: ArangoDB_DGL_Adapter, name: str, vcols: set, ecols: set
):
    assert_adapter_type(adapter)
    dgl_g = adapter.arangodb_collections_to_dgl(
        name,
        vcols,
        ecols,
    )
    assert_dgl_data(dgl_g, vcols, ecols)


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
