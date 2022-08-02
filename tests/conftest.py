import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.http import DefaultHTTPClient
from dgl import DGLGraph, DGLHeteroGraph, heterograph, remove_self_loop
from dgl.data import KarateClubDataset, MiniGCDataset
from pandas import DataFrame
from torch import Tensor, ones, rand, tensor, zeros

from adbdgl_adapter import ADBDGL_Adapter, ADBDGL_Controller
from adbdgl_adapter.typings import DGLCanonicalEType, Json

con: Json
db: StandardDatabase
adbdgl_adapter: ADBDGL_Adapter
PROJECT_DIR = Path(__file__).parent.parent


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="")


def pytest_configure(config: Any) -> None:
    con = {
        "url": config.getoption("url"),
        "username": config.getoption("username"),
        "password": config.getoption("password"),
        "dbName": config.getoption("dbName"),
    }

    print("----------------------------------------")
    print("URL: " + con["url"])
    print("Username: " + con["username"])
    print("Password: " + con["password"])
    print("Database: " + con["dbName"])
    print("----------------------------------------")

    class NoTimeoutHTTPClient(DefaultHTTPClient):  # type: ignore
        REQUEST_TIMEOUT = None

    global db
    db = ArangoClient(hosts=con["url"], http_client=NoTimeoutHTTPClient()).db(
        con["dbName"], con["username"], con["password"], verify=True
    )

    global adbdgl_adapter
    adbdgl_adapter = ADBDGL_Adapter(db, logging_lvl=logging.DEBUG)


def arango_restore(con: Json, path_to_data: str) -> None:
    restore_prefix = "./tools/" if os.getenv("GITHUB_ACTIONS") else ""
    protocol = "http+ssl://" if "https://" in con["url"] else "tcp://"
    url = protocol + con["url"].partition("://")[-1]

    subprocess.check_call(
        f'chmod -R 755 ./tools/arangorestore && {restore_prefix}arangorestore \
            -c none --server.endpoint {url} --server.database {con["dbName"]} \
                --server.username {con["username"]} \
                    --server.password "{con["password"]}" \
                        --input-directory "{PROJECT_DIR}/{path_to_data}"',
        cwd=f"{PROJECT_DIR}/tests",
        shell=True,
    )


def get_karate_graph() -> DGLGraph:
    return KarateClubDataset()[0]


def get_lollipop_graph() -> DGLGraph:
    dgl_g = remove_self_loop(MiniGCDataset(8, 7, 8)[3][0])
    dgl_g.ndata["node_features"] = tensor(
        [[i, i, i] for i in range(0, dgl_g.num_nodes())]
    )
    dgl_g.edata["edge_features"] = rand(dgl_g.num_edges())
    return dgl_g


def get_hypercube_graph() -> DGLGraph:
    dgl_g = remove_self_loop(MiniGCDataset(8, 8, 9)[4][0])
    dgl_g.ndata["node_features"] = rand(dgl_g.num_nodes())
    dgl_g.edata["edge_features"] = tensor(
        [[[i], [i], [i]] for i in range(0, dgl_g.num_edges())]
    )
    return dgl_g


def get_clique_graph() -> DGLGraph:
    dgl_g = remove_self_loop(MiniGCDataset(8, 6, 7)[6][0])
    dgl_g.ndata["node_features"] = ones(dgl_g.num_nodes())
    dgl_g.edata["edge_features"] = zeros(dgl_g.num_edges())
    return dgl_g


def get_fake_hetero_dataset() -> DGLHeteroGraph:
    data_dict = {
        ("v0", "e0", "v0"): (tensor([0, 1, 2, 3, 4, 5]), tensor([5, 4, 3, 2, 1, 0])),
        ("v0", "e0", "v1"): (tensor([0, 1, 2, 3, 4, 5]), tensor([0, 5, 1, 4, 2, 3])),
        ("v0", "e0", "v2"): (tensor([0, 1, 2, 3, 4, 5]), tensor([1, 1, 1, 5, 5, 5])),
        ("v1", "e0", "v1"): (tensor([0, 1, 2, 3, 4, 5]), tensor([3, 3, 3, 3, 3, 3])),
        ("v1", "e0", "v2"): (tensor([0, 1, 2, 3, 4, 5]), tensor([0, 1, 2, 3, 4, 5])),
        ("v2", "e0", "v2"): (tensor([0, 1, 2, 3, 4, 5]), tensor([5, 4, 3, 2, 1, 0])),
    }

    dgl_g: DGLHeteroGraph = heterograph(data_dict)
    dgl_g.nodes["v0"].data["features"] = rand(6)
    dgl_g.nodes["v0"].data["label"] = tensor([1, 3, 2, 1, 3, 2])
    dgl_g.nodes["v1"].data["features"] = rand(6, 1)
    dgl_g.nodes["v2"].data["features"] = rand(6, 2)
    dgl_g.edata["features"] = {("v0", "e0", "v0"): rand(6, 3)}

    return dgl_g


def get_social_graph() -> DGLHeteroGraph:
    dgl_g = heterograph(
        {
            ("user", "follows", "user"): (tensor([0, 1]), tensor([1, 2])),
            ("user", "follows", "game"): (tensor([0, 1, 2]), tensor([0, 1, 2])),
            ("user", "plays", "game"): (tensor([3, 3]), tensor([1, 2])),
        }
    )

    dgl_g.nodes["user"].data["node_features"] = tensor(
        [[21, 0], [16, 1], [38, 0], [64, 0]]
    )
    dgl_g.edges["plays"].data["edge_features"] = tensor([3, 5])

    return dgl_g


# For DGL to ArangoDB testing purposes
def udf_users_features_tensor_to_df(t: Tensor) -> DataFrame:
    df = DataFrame(columns=["age", "gender"])
    df[["age", "gender"]] = t.tolist()
    df["gender"] = df["gender"].map({0: "Male", 1: "Female"})
    return df


# For ArangoDB to DGL testing purposes
def udf_node_features_df_to_tensor(df: DataFrame) -> Tensor:
    return tensor(df["node_features"].to_list())


# For ArangoDB to DGL testing purposes
def udf_key_df_to_tensor(key: str) -> Callable[[DataFrame], Tensor]:
    def f(df: DataFrame) -> Tensor:
        return tensor(df[key].to_list())

    return f


class Custom_ADBDGL_Controller(ADBDGL_Controller):
    def _prepare_dgl_node(self, dgl_node: Json, node_type: str) -> Json:
        dgl_node["foo"] = "bar"
        return dgl_node

    def _prepare_dgl_edge(self, dgl_edge: Json, edge_type: DGLCanonicalEType) -> Json:
        dgl_edge["bar"] = "foo"
        return dgl_edge
