from typing import Union

import pytest
from conftest import (
    ArangoDB_DGL_Adapter,
    get_karate_graph,
    get_lollipop_graph,
    get_hypercube_graph,
    get_clique_graph,
    db,
    conn,
    adbdgl_adapter,
)

from dgl import DGLGraph
from dgl.heterograph import DGLHeteroGraph
from arango.graph import Graph as ArangoGraph

import torch
from torch.functional import Tensor


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
    "adapter, name, metagraph",
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
def test_adb_to_dgl(adapter: ArangoDB_DGL_Adapter, name: str, metagraph: dict):
    assert_adapter_type(adapter)
    dgl_g = adapter.arangodb_to_dgl(name, metagraph)
    assert_dgl_data(dgl_g, metagraph["vertexCollections"], metagraph["edgeCollections"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, name, v_cols, e_cols",
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
    adapter: ArangoDB_DGL_Adapter, name: str, v_cols: set, e_cols: set
):
    assert_adapter_type(adapter)
    dgl_g = adapter.arangodb_collections_to_dgl(
        name,
        v_cols,
        e_cols,
    )
    assert_dgl_data(
        dgl_g, {v_col: {} for v_col in v_cols}, {e_col: {} for e_col in e_cols}
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, name",
    [(adbdgl_adapter, "fraud-detection")],
)
def test_adb_graph_to_dgl(adapter: ArangoDB_DGL_Adapter, name: str):
    assert_adapter_type(adapter)

    arango_graph = db.graph(name)
    v_cols = arango_graph.vertex_collections()
    e_cols = {col["edge_collection"] for col in arango_graph.edge_definitions()}

    dgl_g: DGLGraph = adapter.arangodb_graph_to_dgl(name)
    assert_dgl_data(
        dgl_g, {v_col: {} for v_col in v_cols}, {e_col: {} for e_col in e_cols}
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter, name, dgl_g, is_dgl_data, batch_size",
    [
        (adbdgl_adapter, "Clique", get_clique_graph(), True, 3),
        (adbdgl_adapter, "Lollipop", get_lollipop_graph(), True, 1000),
        (adbdgl_adapter, "Hypercube", get_hypercube_graph(), True, 1000),
        (adbdgl_adapter, "Karate", get_karate_graph(), True, 1000),
    ],
)
def test_dgl_to_adb(
    adapter: ArangoDB_DGL_Adapter,
    name: str,
    dgl_g: Union[DGLGraph, DGLHeteroGraph],
    is_dgl_data: bool,
    batch_size: int,
):
    assert_adapter_type(adapter)
    adb_g = adapter.dgl_to_arangodb(name, dgl_g, batch_size)
    assert_arangodb_data(name, dgl_g, adb_g, is_dgl_data)


def assert_adapter_type(adapter: ArangoDB_DGL_Adapter):
    assert type(adapter) is ArangoDB_DGL_Adapter


def assert_dgl_data(dgl_g: DGLGraph, v_cols: dict, e_cols: dict):
    for col, atribs in v_cols.items():
        num_nodes = dgl_g.num_nodes(col)
        assert num_nodes == db.collection(col).count()

        for atrib in atribs:
            assert atrib in dgl_g.ndata
            assert col in dgl_g.ndata[atrib]
            assert len(dgl_g.ndata[atrib][col]) == num_nodes

    for col, atribs in e_cols.items():
        num_edges = dgl_g.num_edges(col)
        assert num_edges == db.collection(col).count()

        canon_etype = dgl_g.to_canonical_etype(col)
        for atrib in atribs:
            assert atrib in dgl_g.edata
            assert canon_etype in dgl_g.edata[atrib]
            assert len(dgl_g.edata[atrib][canon_etype]) == num_edges


def assert_arangodb_data(
    name: str,
    dgl_g: Union[DGLGraph, DGLHeteroGraph],
    adb_g: ArangoGraph,
    is_dgl_data: bool,
):
    for dgl_v_col in dgl_g.ntypes:
        adb_v_col = name + dgl_v_col if is_dgl_data else dgl_v_col
        attributes = dgl_g.node_attr_schemes(None if is_dgl_data else dgl_v_col).keys()
        col = adb_g.vertex_collection(adb_v_col)

        node: Tensor
        for node in dgl_g.nodes(dgl_v_col):
            vertex = col.get(str(node.item()))
            assert vertex
            for atrib in attributes:
                assert atrib in vertex

    for dgl_e_col in dgl_g.etypes:
        dgl_from_col, _, dgl_to_col = dgl_g.to_canonical_etype(dgl_e_col)
        attributes = dgl_g.edge_attr_schemes(None if is_dgl_data else dgl_e_col).keys()

        adb_e_col = name + dgl_e_col if is_dgl_data else dgl_e_col
        adb_from_col = name + dgl_v_col if is_dgl_data else dgl_from_col
        adb_to_col = name + dgl_v_col if is_dgl_data else dgl_to_col

        col = adb_g.edge_collection(adb_e_col)

        from_node: Tensor
        to_node: Tensor
        from_nodes, to_nodes = dgl_g.edges(etype=dgl_e_col)
        for from_node, to_node in zip(from_nodes, to_nodes):
            edge = col.find(
                {
                    "_from": f"{adb_from_col}/{str(from_node.item())}",
                    "_to": f"{adb_to_col}/{str(to_node.item())}",
                }
            ).next()
            assert edge
            for atrib in attributes:
                assert atrib in edge
