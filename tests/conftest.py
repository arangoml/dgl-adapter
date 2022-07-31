import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from arango import ArangoClient
from arango.database import StandardDatabase
from dgl import DGLGraph, heterograph, remove_self_loop
from dgl.data import KarateClubDataset, MiniGCDataset
from torch import ones, rand, tensor, zeros

from adbdgl_adapter import ADBDGL_Adapter
from adbdgl_adapter.typings import Json

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

    global db
    db = ArangoClient(hosts=con["url"]).db(
        con["dbName"], con["username"], con["password"], verify=True
    )

    global adbdgl_adapter
    adbdgl_adapter = ADBDGL_Adapter(db, logging_lvl=logging.DEBUG)

    if db.has_graph("fraud-detection") is False:
        arango_restore(con, "examples/data/fraud_dump")
        db.create_graph(
            "fraud-detection",
            edge_definitions=[
                {
                    "edge_collection": "accountHolder",
                    "from_vertex_collections": ["customer"],
                    "to_vertex_collections": ["account"],
                },
                {
                    "edge_collection": "transaction",
                    "from_vertex_collections": ["account"],
                    "to_vertex_collections": ["account"],
                },
            ],
        )


def arango_restore(con: Json, path_to_data: str) -> None:
    restore_prefix = "./assets/" if os.getenv("GITHUB_ACTIONS") else ""
    protocol = "http+ssl://" if "https://" in con["url"] else "tcp://"
    url = protocol + con["url"].partition("://")[-1]

    subprocess.check_call(
        f'chmod -R 755 ./assets/arangorestore && {restore_prefix}arangorestore \
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
    dgl_g.ndata["random_ndata"] = tensor(
        [[i, i, i] for i in range(0, dgl_g.num_nodes())]
    )
    dgl_g.edata["random_edata"] = rand(dgl_g.num_edges())
    return dgl_g


def get_hypercube_graph() -> DGLGraph:
    dgl_g = remove_self_loop(MiniGCDataset(8, 8, 9)[4][0])
    dgl_g.ndata["random_ndata"] = rand(dgl_g.num_nodes())
    dgl_g.edata["random_edata"] = tensor(
        [[[i], [i], [i]] for i in range(0, dgl_g.num_edges())]
    )
    return dgl_g


def get_clique_graph() -> DGLGraph:
    dgl_g = remove_self_loop(MiniGCDataset(8, 6, 7)[6][0])
    dgl_g.ndata["random_ndata"] = ones(dgl_g.num_nodes())
    dgl_g.edata["random_edata"] = zeros(dgl_g.num_edges())
    return dgl_g


def get_social_graph() -> DGLGraph:
    dgl_g = heterograph(
        {
            ("user", "follows", "user"): (tensor([0, 1]), tensor([1, 2])),
            ("user", "follows", "game"): (tensor([0, 1, 2]), tensor([0, 1, 2])),
            ("user", "plays", "game"): (tensor([3, 3]), tensor([1, 2])),
        }
    )

    dgl_g.nodes["user"].data["age"] = tensor([21, 16, 38, 64])
    dgl_g.edges["plays"].data["hours_played"] = tensor([3, 5])

    return dgl_g
