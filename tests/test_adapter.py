from typing import Any, Dict, Set, Union

import pytest
from arango.database import StandardDatabase
from arango.graph import Graph as ArangoGraph
from dgl import DGLGraph
from dgl.heterograph import DGLHeteroGraph
from torch.functional import Tensor

from adbdgl_adapter import ADBDGL_Adapter
from adbdgl_adapter.typings import ArangoMetagraph

from .conftest import (
    adbdgl_adapter,
    db,
    get_clique_graph,
    get_hypercube_graph,
    get_karate_graph,
    get_lollipop_graph,
    get_social_graph,
)


def test_validate_constructor() -> None:
    bad_db: Dict[str, Any] = dict()

    class Bad_ADBDGL_Controller:
        pass

    with pytest.raises(TypeError):
        ADBDGL_Adapter(bad_db)

    with pytest.raises(TypeError):
        ADBDGL_Adapter(db, Bad_ADBDGL_Controller())  # type: ignore


@pytest.mark.parametrize(
    "adapter, name, metagraph",
    [
        (
            adbdgl_adapter,
            "fraud-detection",
            {
                "vertexCollections": {
                    "account": {"Balance", "rank"},
                    "customer": {"rank"},
                    "Class": {},
                },
                "edgeCollections": {
                    "transaction": {
                        "transaction_amt",
                        "sender_bank_id",
                        "receiver_bank_id",
                    },
                    "accountHolder": {},
                    "Relationship": {},
                },
            },
        ),
    ],
)
def test_adb_to_dgl(
    adapter: ADBDGL_Adapter, name: str, metagraph: ArangoMetagraph
) -> None:
    dgl_g = adapter.arangodb_to_dgl(name, metagraph)
    assert_dgl_data(adapter.db, dgl_g, metagraph)
    print(dgl_g.edata)


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
    adapter: ADBDGL_Adapter, name: str, v_cols: Set[str], e_cols: Set[str]
) -> None:
    dgl_g = adapter.arangodb_collections_to_dgl(
        name,
        v_cols,
        e_cols,
    )
    assert_dgl_data(
        adapter.db,
        dgl_g,
        metagraph={
            "vertexCollections": {col: set() for col in v_cols},
            "edgeCollections": {col: set() for col in e_cols},
        },
    )


@pytest.mark.parametrize(
    "adapter, name",
    [(adbdgl_adapter, "fraud-detection")],
)
def test_adb_graph_to_dgl(adapter: ADBDGL_Adapter, name: str) -> None:
    arango_graph = db.graph(name)
    v_cols = arango_graph.vertex_collections()
    e_cols = {col["edge_collection"] for col in arango_graph.edge_definitions()}

    dgl_g: DGLGraph = adapter.arangodb_graph_to_dgl(name)
    assert_dgl_data(
        adapter.db,
        dgl_g,
        metagraph={
            "vertexCollections": {col: set() for col in v_cols},
            "edgeCollections": {col: set() for col in e_cols},
        },
    )


@pytest.mark.parametrize(
    "adapter, name, dgl_g, overwrite_graph, import_options",
    [
        (
            adbdgl_adapter,
            "Clique",
            get_clique_graph(),
            False,
            {"batch_size": 3, "on_duplicate": "replace"},
        ),
        (adbdgl_adapter, "Lollipop", get_lollipop_graph(), False, {"overwrite": True}),
        (
            adbdgl_adapter,
            "Hypercube",
            get_hypercube_graph(),
            False,
            {"batch_size": 1000, "on_duplicate": "replace"},
        ),
        (
            adbdgl_adapter,
            "Hypercube",
            get_hypercube_graph(),
            False,
            {"overwrite": True},
        ),
        (adbdgl_adapter, "Karate", get_karate_graph(), False, {"overwrite": True}),
        (adbdgl_adapter, "Social", get_social_graph(), True, {"on_duplicate": "replace"}),
    ],
)
def test_dgl_to_adb(
    adapter: ADBDGL_Adapter,
    name: str,
    dgl_g: Union[DGLGraph, DGLHeteroGraph],
    overwrite_graph: bool,
    import_options: Any,
) -> None:
    adb_g = adapter.dgl_to_arangodb(name, dgl_g, overwrite_graph, **import_options)
    assert_arangodb_data(name, dgl_g, adb_g)


def assert_dgl_data(
    db: StandardDatabase, dgl_g: DGLGraph, metagraph: ArangoMetagraph
) -> None:
    has_one_ntype = len(metagraph["vertexCollections"]) == 1
    has_one_etype = len(metagraph["edgeCollections"]) == 1

    for col, atribs in metagraph["vertexCollections"].items():
        num_nodes = dgl_g.num_nodes(col)
        assert num_nodes == db.collection(col).count()

        for atrib in atribs:
            assert atrib in dgl_g.ndata
            if has_one_ntype:
                assert len(dgl_g.ndata[atrib]) == num_nodes
            else:
                assert col in dgl_g.ndata[atrib]
                assert len(dgl_g.ndata[atrib][col]) == num_nodes

    for col, atribs in metagraph["edgeCollections"].items():
        num_edges = dgl_g.num_edges(col)
        assert num_edges == db.collection(col).count()

        canon_etype = dgl_g.to_canonical_etype(col)
        for atrib in atribs:
            assert atrib in dgl_g.edata
            if has_one_etype:
                assert len(dgl_g.edata[atrib]) == num_edges
            else:
                assert canon_etype in dgl_g.edata[atrib]
                assert len(dgl_g.edata[atrib][canon_etype]) == num_edges


def assert_arangodb_data(
    name: str,
    dgl_g: Union[DGLGraph, DGLHeteroGraph],
    adb_g: ArangoGraph,
) -> None:
    is_default_type = dgl_g.canonical_etypes == adbdgl_adapter.DEFAULT_CANONICAL_ETYPE

    node: Tensor
    for ntype in dgl_g.ntypes:
        adb_v_col = f"{name}_N" if is_default_type else ntype
        attributes = dgl_g.node_attr_schemes(ntype).keys()
        col = adb_g.vertex_collection(adb_v_col)

        for node in dgl_g.nodes(ntype):
            vertex = col.get(str(node.item()))
            assert vertex
            for atrib in attributes:
                assert atrib in vertex

    from_node: Tensor
    to_node: Tensor
    for c_etype in dgl_g.canonical_etypes:
        dgl_from_col, dgl_e_col, dgl_to_col = c_etype
        attributes = dgl_g.edge_attr_schemes(c_etype).keys()

        adb_e_col = f"{name}_E" if is_default_type else dgl_e_col
        adb_from_col = f"{name}_N" if is_default_type else dgl_from_col
        adb_to_col = f"{name}_N" if is_default_type else dgl_to_col

        col = adb_g.edge_collection(adb_e_col)

        from_nodes, to_nodes = dgl_g.edges(etype=c_etype)
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
