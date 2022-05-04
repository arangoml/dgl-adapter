import os
import subprocess
from pathlib import Path
from typing import Any

from dgl import DGLGraph, remove_self_loop
from dgl.data import KarateClubDataset, MiniGCDataset
from torch import ones, rand, tensor, zeros

from adbdgl_adapter.adapter import ADBDGL_Adapter
from adbdgl_adapter.typings import Json

con: Json
adbdgl_adapter: ADBDGL_Adapter
PROJECT_DIR = Path(__file__).parent.parent


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--protocol", action="store", default="http")
    parser.addoption("--host", action="store", default="localhost")
    parser.addoption("--port", action="store", default="8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="openSesame")


def pytest_configure(config: Any) -> None:
    global con
    con = {
        "protocol": config.getoption("protocol"),
        "hostname": config.getoption("host"),
        "port": config.getoption("port"),
        "username": config.getoption("username"),
        "password": config.getoption("password"),
        "dbName": config.getoption("dbName"),
    }

    print("----------------------------------------")
    print(f"{con['protocol']}://{con['hostname']}:{con['port']}")
    print("Username: " + con["username"])
    print("Password: " + con["password"])
    print("Database: " + con["dbName"])
    print("----------------------------------------")

    global adbdgl_adapter
    adbdgl_adapter = ADBDGL_Adapter(con)

    # Restore fraud dataset via arangorestore
    arango_restore(con, "examples/data/fraud_dump")

    # Create Fraud Detection Graph
    adbdgl_adapter.db().delete_graph("fraud-detection", ignore_missing=True)
    adbdgl_adapter.db().create_graph(
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

    subprocess.check_call(
        f'chmod -R 755 ./assets/arangorestore && {restore_prefix}arangorestore \
            -c none --server.endpoint tcp://{con["hostname"]}:{con["port"]} \
                --server.username {con["username"]} --server.database {con["dbName"]} \
                    --server.password {con["password"]} \
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
