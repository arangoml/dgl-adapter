from typing import Any, Dict, List, Optional, Set, Union
from pandas import DataFrame

import pytest
from arango.database import StandardDatabase
from arango.graph import Graph as ArangoGraph
from dgl import DGLGraph, DGLHeteroGraph
from dgl.view import NodeSpace, EdgeSpace
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor, tensor

from adbdgl_adapter import ADBDGL_Adapter
from adbdgl_adapter.encoders import IdentityEncoder
from adbdgl_adapter.exceptions import ADBMetagraphError, DGLMetagraphError
from adbdgl_adapter.typings import *
from adbdgl_adapter.utils import validate_adb_metagraph, validate_dgl_metagraph

from .conftest import (
    adbdgl_adapter,
    db,
    # con,
    arango_restore,
    get_fake_hetero_dataset,
    get_hypercube_graph,
    get_karate_graph,
    get_social_graph,
    label_tensor_to_2_column_dataframe,
    udf_features_df_to_tensor,
    udf_key_df_to_tensor,
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
    "bad_metagraph",
    [  # empty metagraph
        ({}),
        # missing required parent key
        ({"vertexCollections": {}}),
        # empty sub-metagraph
        ({"vertexCollections": {}, "edgeCollections": {}}),
        # bad collection name
        (
            {
                "vertexCollections": {
                    1: {},
                    # other examples include:
                    # True: {},
                    # ('a'): {}
                },
                "edgeCollections": {},
            }
        ),
        # bad collection metagraph
        (
            {
                "vertexCollections": {
                    "vcol_a": None,
                    # other examples include:
                    # "vcol_a": 1,
                    # "vcol_a": 'foo',
                },
                "edgeCollections": {},
            }
        ),
        # bad collection metagraph 2
        (
            {
                "vertexCollections": {
                    "vcol_a": {"a", "b", 3},
                    # other examples include:
                    # "vcol_a": 1,
                    # "vcol_a": 'foo',
                },
                "edgeCollections": {},
            }
        ),
        # bad meta_key
        (
            {
                "vertexCollections": {
                    "vcol_a": {
                        1: {},
                        # other example include:
                        # True: {},
                        # ("x"): {},
                    }
                },
                "edgeCollections": {},
            }
        ),
        # bad meta_val
        (
            {
                "vertexCollections": {
                    "vcol_a": {
                        "x": True,
                        # other example include:
                        # 'x': ('a'),
                        # 'x': ['a'],
                        # 'x': 5
                    }
                },
                "edgeCollections": {},
            }
        ),
        # bad meta_val encoder key
        (
            {
                "vertexCollections": {"vcol_a": {"x": {1: IdentityEncoder()}}},
                "edgeCollections": {},
            }
        ),
        # bad meta_val encoder value
        (
            {
                "vertexCollections": {
                    "vcol_a": {
                        "x": {
                            "Action": True,
                            # other examples include:
                            # 'Action': {}
                            # 'Action': (lambda : 1)()
                        }
                    }
                },
                "edgeCollections": {},
            }
        ),
    ],
)
def test_validate_adb_metagraph(bad_metagraph: Dict[Any, Any]) -> None:
    with pytest.raises(ADBMetagraphError):
        validate_adb_metagraph(bad_metagraph)


@pytest.mark.parametrize(
    "bad_metagraph",
    [
        # bad node type
        (
            {
                "nodeTypes": {
                    ("a", "b", "c"): {},
                    # other examples include:
                    # 1: {},
                    # True: {}
                }
            }
        ),
        # bad edge type
        (
            {
                "edgeTypes": {
                    "b": {},
                    # other examples include:
                    # 1: {},
                    # True: {}
                }
            }
        ),
        # bad edge type 2
        (
            {
                "edgeTypes": {
                    ("a", "b", 3): {},
                    # other examples include:
                    # 1: {},
                    # True: {}
                }
            }
        ),
        # bad data type metagraph
        (
            {
                "nodeTypes": {
                    "ntype_a": None,
                    # other examples include:
                    # "ntype_a": 1,
                    # "ntype_a": 'foo',
                }
            }
        ),
        # bad data type metagraph 2
        (
            {
                "nodeTypes": {
                    "ntype_a": {"a", "b", 3}
                }
            }
        ),
        # bad meta_val
        (
            {
                "nodeTypes": {
                    "ntype_a'": {
                        "x": True,
                        # other example include:
                        # 'x': ('a'),
                        # 'x': (lambda: 1)(),
                    }
                }
            }
        ),
        # bad meta_val list
        (
            {
                "nodeTypes": {
                    "ntype_a'": {
                        "x": ["a", 3],
                        # other example include:
                        # 'x': ('a'),
                        # 'x': (lambda: 1)(),
                    }
                }
            }
        ),
    ],
)
def test_validate_dgl_metagraph(bad_metagraph: Dict[Any, Any]) -> None:
    with pytest.raises(DGLMetagraphError):
        validate_dgl_metagraph(bad_metagraph)


@pytest.mark.parametrize(
    "adapter, name, dgl_g, metagraph, \
        explicit_metagraph, overwrite_graph, import_options",
    [
        (
            adbdgl_adapter,
            "Karate_2",
            get_karate_graph(),
            {"nodeTypes": {"Karate_1_N": {"label": "node_label"}}},
            False,
            False,
            {},
        ),
        (
            adbdgl_adapter,
            "Karate_2",
            get_karate_graph(),
            {"nodeTypes": {"Karate_2_N": {}}},
            True,
            False,
            {},
        ),
        (
            adbdgl_adapter,
            "Social_1",
            get_social_graph(),
            {
                "nodeTypes": {
                    "user": {
                        "features": "user_age",
                        "label": label_tensor_to_2_column_dataframe,
                    },
                    "game": {"features": ["is_multiplayer", "is_free_to_play"]},
                },
                "edgeTypes": {
                    ("user", "plays", "game"): {
                        "features": ["hours_played", "is_satisfied_with_game"]
                    },
                },
            },
            True,
            False,
            {},
        ),
        (
            adbdgl_adapter,
            "Social_2",
            get_social_graph(),
            {
                "edgeTypes": {
                    ("user", "plays", "game"): {
                        "features": ["hours_played", "is_satisfied_with_game"]
                    },
                },
            },
            True,
            False,
            {},
        ),
        (
            adbdgl_adapter,
            "Social_3",
            get_social_graph(),
            {},
            False,
            False,
            {},
        ),
        (
            adbdgl_adapter,
            "FakeHeterogeneous_1",
            get_fake_hetero_dataset(),
            {
                "nodeTypes": {
                    "v0": {"features": "adb_node_features", "label": "adb_node_label"}
                },
                "edgeTypes": {("v0", "e0", "v0"): {"features": "adb_edge_features"}},
            },
            True,
            False,
            {},
        ),
        (
            adbdgl_adapter,
            "FakeHeterogeneous_2",
            get_fake_hetero_dataset(),
            {},
            False,
            False,
            {},
        ),
        (
            adbdgl_adapter,
            "FakeHeterogeneous_3",
            get_fake_hetero_dataset(),
            {
                "nodeTypes": {
                    "v0": {"features", "label"}
                },
                "edgeTypes": {("v0", "e0", "v0"): {"features"}},
            },
            True,
            False,
            {},
        ),
    ],
)
def test_dgl_to_adb(
    adapter: ADBDGL_Adapter,
    name: str,
    dgl_g: Union[DGLGraph, DGLHeteroGraph],
    metagraph: DGLMetagraph,
    explicit_metagraph: bool,
    overwrite_graph: bool,
    import_options: Any,
) -> None:
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adapter.dgl_to_arangodb(
        name, dgl_g, metagraph, explicit_metagraph, overwrite_graph, **import_options
    )
    assert_dgl_to_adb(name, dgl_g, metagraph, explicit_metagraph)
    db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize(
    "adapter, name, metagraph, dgl_g_old",
    [
        (
            adbdgl_adapter,
            "Karate",
            {
                "vertexCollections": {
                    "Karate_N": {"node_features": "node_features", "label": "label"},
                },
                "edgeCollections": {
                    "Karate_E": {},
                },
            },
            get_karate_graph(),
        ),
        (
            adbdgl_adapter,
            "Hypercube",
            {
                "vertexCollections": {
                    "Hypercube_N": {"node_features": "node_features"},
                },
                "edgeCollections": {
                    "Hypercube_E": {"edge_features": "edge_features"},
                },
            },
            get_hypercube_graph(),
        ),
        (
            adbdgl_adapter,
            "Social",
            {
                "vertexCollections": {
                    "user": {"features": "node_features", "label": "label"},
                    "game": {"features": "node_features"},
                    "topic": {},
                },
                "edgeCollections": {
                    "plays": {"features": "edge_features"},
                    "follows": {},
                },
            },
            get_social_graph(),
        ),
        (
            adbdgl_adapter,
            "Heterogeneous",
            {
                "vertexCollections": {
                    "v0": {"features": "features", "label": "label"},
                    "v1": {"features": "features"},
                    "v2": {"features": "features"},
                },
                "edgeCollections": {
                    "e0": {},
                },
            },
            get_fake_hetero_dataset(),
        ),
        (
            adbdgl_adapter,
            "HeterogeneousOverComplicatedMetagraph",
            {
                "vertexCollections": {
                    "v0": {"features": {"features": None}, "label": {"label": None}},
                    "v1": {"features": "features"},
                    "v2": {"features": {"features": None}},
                },
                "edgeCollections": {
                    "e0": {},
                },
            },
            get_fake_hetero_dataset(),
        ),
        (
            adbdgl_adapter,
            "HeterogeneousUserDefinedFunctions",
            {
                "vertexCollections": {
                    "v0": {
                        "features": (lambda df: tensor(df["features"].to_list())),
                        "label": (lambda df: tensor(df["label"].to_list())),
                    },
                    "v1": {"features": udf_features_df_to_tensor},
                    "v2": {"features": udf_key_df_to_tensor("features")},
                },
                "edgeCollections": {
                    "e0": {},
                },
            },
            get_fake_hetero_dataset(),
        ),
    ],
)
def test_adb_to_pyg(
    adapter: ADBDGL_Adapter,
    name: str,
    metagraph: ADBMetagraph,
    dgl_g_old: Optional[Union[DGLGraph, DGLHeteroGraph]],
) -> None:
    if dgl_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.dgl_to_arangodb(name, dgl_g_old)

    dgl_g_new = adapter.arangodb_to_dgl(name, metagraph)
    assert_adb_to_dgl(dgl_g_new, metagraph)

    if dgl_g_old:
        db.delete_graph(name, drop_collections=True)


def test_adb_partial_to_pyg() -> None:
    dgl_g = get_social_graph()

    name = "Social"
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adbdgl_adapter.dgl_to_arangodb(name, dgl_g)

    metagraph: ADBMetagraph

    # Case 1: Partial edge collection import turns the graph homogeneous
    metagraph = {
        "vertexCollections": {
            "user": {"features": "features", "label": "label"},
        },
        "edgeCollections": {
            "follows": {},
        },
    }

    dgl_g_new = adbdgl_adapter.arangodb_to_dgl(
        "HeterogeneousTurnedHomogeneous", metagraph
    )

    assert type(dgl_g_new) is DGLGraph
    assert (
        dgl_g.ndata["features"]["user"].tolist() == dgl_g_new.ndata["features"].tolist()
    )
    assert dgl_g.ndata["label"]["user"].tolist() == dgl_g_new["label"].tolist()

    from_nodes, to_nodes = dgl_g.edges(
        etype=("user", "follows", "user")
    )  # Heterogeneous
    from_nodes_new, to_nodes_new = dgl_g_new.edges(etype=None)  # Homogeneous
    assert from_nodes.tolist() == from_nodes_new.tolist()
    assert to_nodes.tolist() == to_nodes.new.tolist()

    # Case 2: Partial edge collection import keeps the graph heterogeneous
    metagraph = {
        "vertexCollections": {
            "user": {"features": "features", "label": "label"},
            "game": {"features": "features"},
        },
        "edgeCollections": {"follows": {}, "plays": {"features": "features"}},
    }

    dgl_g_new = adbdgl_adapter.arangodb_to_dgl(
        "HeterogeneousWithOneLessNodeType", metagraph
    )

    assert type(dgl_g_new) is DGLHeteroGraph
    assert set(dgl_g_new.ntypes) == {"user", "game"}
    for n_type in dgl_g_new.ntypes:
        for k, v in dgl_g_new.nodes[n_type].items():
            assert v.tolist() == dgl_g.nodes[n_type][k].tolist()

    for e_type in dgl_g_new.canonical_etypes:
        for k, v in dgl_g_new.edges[e_type].items():
            assert v.tolist() == dgl_g.edges[e_type][k].tolist()

    db.delete_graph(name, drop_collections=True)


# @pytest.mark.parametrize(
#     "adapter, name, v_cols, e_cols",
#     [
#         (
#             adbdgl_adapter,
#             "fraud-detection",
#             {"account", "Class", "customer"},
#             {"accountHolder", "Relationship", "transaction"},
#         )
#     ],
# )
# def test_adb_collections_to_dgl(
#     adapter: ADBDGL_Adapter, name: str, v_cols: Set[str], e_cols: Set[str]
# ) -> None:
#     dgl_g = adapter.arangodb_collections_to_dgl(
#         name,
#         v_cols,
#         e_cols,
#     )
#     assert_dgl_data(
#         db,
#         dgl_g,
#         metagraph={
#             "vertexCollections": {col: set() for col in v_cols},
#             "edgeCollections": {col: set() for col in e_cols},
#         },
#     )


# @pytest.mark.parametrize(
#     "adapter, name",
#     [(adbdgl_adapter, "fraud-detection")],
# )
# def test_adb_graph_to_dgl(adapter: ADBDGL_Adapter, name: str) -> None:
#     arango_graph = db.graph(name)
#     v_cols = arango_graph.vertex_collections()
#     e_cols = {col["edge_collection"] for col in arango_graph.edge_definitions()}

#     dgl_g: DGLGraph = adapter.arangodb_graph_to_dgl(name)
#     assert_dgl_data(
#         db,
#         dgl_g,
#         metagraph={
#             "vertexCollections": {col: set() for col in v_cols},
#             "edgeCollections": {col: set() for col in e_cols},
#         },
#     )


@pytest.mark.parametrize(
    "adapter, name, v_cols, e_cols, dgl_g_old",
    [(adbdgl_adapter, "Social", {"user", "game"}, {"plays"}, get_social_graph())],
)
def test_adb_collections_to_pyg(
    adapter: ADBDGL_Adapter,
    name: str,
    v_cols: Set[str],
    e_cols: Set[str],
    dgl_g_old: Union[DGLGraph, DGLHeteroGraph],
) -> None:
    if dgl_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.pyg_to_arangodb(name, dgl_g_old)

    dgl_g_new = adapter.arangodb_collections_to_dgl(
        name,
        v_cols,
        e_cols,
    )

    assert_adb_to_dgl(
        dgl_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    if dgl_g_old:
        db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize(
    "adapter, name, dgl_g_old",
    [
        (adbdgl_adapter, "Heterogeneous", get_fake_hetero_dataset()),
    ],
)
def test_adb_graph_to_pyg(
    adapter: ADBDGL_Adapter, name: str, dgl_g_old: Union[DGLGraph, DGLHeteroGraph]
) -> None:
    if dgl_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.dgl_to_arangodb(name, dgl_g_old)

    dgl_g_new = adapter.arangodb_graph_to_dgl(name)

    arango_graph = db.graph(name)
    v_cols = arango_graph.vertex_collections()
    e_cols = {col["edge_collection"] for col in arango_graph.edge_definitions()}

    assert_adb_to_dgl(
        dgl_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    if dgl_g_old:
        db.delete_graph(name, drop_collections=True)


def test_full_cycle_imdb_without_preserve_adb_keys() -> None:
    name = "imdb"
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    arango_restore(con, "tests/data/adb/imdb_dump")
    db.create_graph(
        name,
        edge_definitions=[
            {
                "edge_collection": "Ratings",
                "from_vertex_collections": ["Users"],
                "to_vertex_collections": ["Movies"],
            },
        ],
    )

    adb_to_pyg_metagraph: ADBMetagraph = {
        "vertexCollections": {
            "Movies": {
                "y": "Comedy",
                "x": {
                    "Action": IdentityEncoder(dtype=long),
                    "Drama": IdentityEncoder(dtype=long),
                    # etc....
                },
            },
            "Users": {
                "x": {
                    "Age": IdentityEncoder(dtype=long),
                    "Gender": CategoricalEncoder(),
                }
            },
        },
        "edgeCollections": {"Ratings": {"edge_weight": "Rating"}},
    }

    pyg_g = adbpyg_adapter.arangodb_to_pyg(name, adb_to_pyg_metagraph)
    assert_adb_to_pyg(pyg_g, adb_to_pyg_metagraph)

    pyg_to_adb_metagraph: PyGMetagraph = {
        "nodeTypes": {
            "Movies": {
                "y": "comedy",
                "x": ["action", "drama"],
            },
            "Users": {"x": udf_users_x_tensor_to_df},
        },
        "edgeTypes": {("Users", "Ratings", "Movies"): {"edge_weight": "rating"}},
    }
    adbpyg_adapter.pyg_to_arangodb(name, pyg_g, pyg_to_adb_metagraph, overwrite=True)
    assert_pyg_to_adb(name, pyg_g, pyg_to_adb_metagraph)

    db.delete_graph(name, drop_collections=True)


def assert_dgl_to_adb(
    name: str,
    dgl_g: Union[DGLGraph, DGLHeteroGraph],
    metagraph: DGLMetagraph,
    explicit_metagraph: bool = False,
) -> None:

    has_one_ntype = len(dgl_g.ntypes) == 1
    has_one_etype = len(dgl_g.canonical_etypes) == 1
    has_default_canonical_etypes = dgl_g.canonical_etypes == [("_N", "_E", "_N")]

    node_types: List[str]
    edge_types: List[DGLCanonicalEType]
    explicit_metagraph = metagraph != {} and explicit_metagraph
    if explicit_metagraph:
        node_types = metagraph.get("nodeTypes", {}).keys()  # type: ignore
        edge_types = metagraph.get("edgeTypes", {}).keys()  # type: ignore

    elif has_default_canonical_etypes:
        n_type = name + "_N"
        node_types = [n_type]
        edge_types = [(n_type, name + "_E", n_type)]

    else:
        node_types = dgl_g.ntypes
        edge_types = dgl_g.canonical_etypes

    n_meta = metagraph.get("nodeTypes", {})
    for n_type in node_types:
        n_key = None if has_one_ntype else n_type
        collection = db.collection(n_type)
        assert collection.count() == dgl_g.num_nodes(n_key)

        df = DataFrame(collection.all())
        meta = n_meta.get(n_type, {})
        assert_dgl_to_adb_meta(df, meta, dgl_g.nodes[n_key].data, explicit_metagraph)

    e_meta = metagraph.get("edgeTypes", {})
    for e_type in edge_types:
        e_key = None if has_one_etype else e_type
        from_col, e_col, to_col = e_type
        collection = db.collection(e_col)

        df = DataFrame(collection.all())
        df[["from_col", "from_key"]] = df["_from"].str.split("/", 1, True)
        df[["to_col", "to_key"]] = df["_to"].str.split("/", 1, True)
        et_df = df[(df["from_col"] == from_col) & (df["to_col"] == to_col)]

        aql = f"""
            FOR edge IN {e_col}
                FILTER IS_SAME_COLLECTION({from_col}, edge._from)
                AND IS_SAME_COLLECTION({to_col}, edge._to)
                RETURN 1
        """

        assert db.aql.execute(aql, count=True).count() == dgl_g.num_edges(e_key)

        meta = e_meta.get(e_type, {})
        assert_dgl_to_adb_meta(et_df, meta, dgl_g.edges[e_key].data, explicit_metagraph)


def assert_dgl_to_adb_meta(
    df: DataFrame,
    meta: Union[str, Dict[Any, DGLMetagraphValues]],
    dgl_data: Union[NodeSpace, EdgeSpace],
    explicit_metagraph: bool,
) -> None:
    valid_meta: Dict[Any, DGLMetagraphValues]
    valid_meta = meta if type(meta) is dict else {m: m for m in meta}   

    if explicit_metagraph:
        dgl_keys = set(valid_meta.keys())
    else:
        dgl_keys = dgl_data.keys()

    for k in dgl_keys:
        data = dgl_data[k]
        meta_val = valid_meta.get(k, str(k))

        assert len(data) == len(df)

        if type(data) is Tensor:
            if type(meta_val) is str:
                assert meta_val in df
                assert df[meta_val].tolist() == data.tolist()

            if type(meta_val) is list:
                assert all([e in df for e in meta_val])
                assert df[meta_val].values.tolist() == data.tolist()

            if callable(meta_val):
                udf_df = meta_val(data)
                assert all([column in df for column in udf_df.columns])
                for column in udf_df.columns:
                    assert df[column].tolist() == udf_df[column].tolist()
