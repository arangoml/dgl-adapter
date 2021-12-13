import io
import os
import time
import json
import zipfile
import requests
import subprocess
from pathlib import Path
import urllib.request as urllib

import dgl
from dgl.data import KarateClubDataset
from dgl.data import MiniGCDataset

import torch
from arango import ArangoClient
from adbdgl_adapter.adbdgl_adapter import ArangoDB_DGL_Adapter

PROJECT_DIR = Path(__file__).parent.parent.parent


def pytest_sessionstart():
    global conn
    conn = get_oasis_crendetials()
    # conn = {
    #     "username": "root",
    #     "password": "openSesame",
    #     "hostname": "localhost",
    #     "port": 8529,
    #     "protocol": "http",
    #     "dbName": "_system",
    # }
    print_connection_details(conn)
    time.sleep(5)  # Enough for the oasis instance to be ready.

    global adbdgl_adapter
    adbdgl_adapter = ArangoDB_DGL_Adapter(conn)

    arango_restore("adbdgl_adapter/tests/data/fraud_dump")

    edge_definitions = [
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
    ]

    global db
    url = (
        conn.get("protocol", "https")
        + "://"
        + conn["hostname"]
        + ":"
        + str(conn["port"])
    )
    db = ArangoClient(hosts=url).db(
        conn["dbName"], conn["username"], conn["password"], verify=True
    )
    db.create_graph("fraud-detection", edge_definitions=edge_definitions)


def get_oasis_crendetials() -> dict:
    url = "https://tutorials.arangodb.cloud:8529/_db/_system/tutorialDB/tutorialDB"
    request = requests.post(url, data=json.dumps("{}"))
    if request.status_code != 200:
        raise Exception("Error retrieving login data.")

    return json.loads(request.text)


def arango_restore(path_to_data):
    restore_prefix = "./" if os.getenv("GITHUB_ACTIONS") else ""  # temporary hack

    subprocess.check_call(
        f'chmod -R 755 ./arangorestore && {restore_prefix}arangorestore -c none --server.endpoint http+ssl://{conn["hostname"]}:{conn["port"]} --server.username {conn["username"]} --server.database {conn["dbName"]} --server.password {conn["password"]} --default-replication-factor 3  --input-directory "{PROJECT_DIR}/{path_to_data}"',
        cwd=f"{PROJECT_DIR}/adbdgl_adapter/tests",
        shell=True,
    )


def print_connection_details(conn):
    print("----------------------------------------")
    print("https://{}:{}".format(conn["hostname"], conn["port"]))
    print("Username: " + conn["username"])
    print("Password: " + conn["password"])
    print("Database: " + conn["dbName"])
    print("----------------------------------------")


def get_karate_graph():
    return KarateClubDataset()[0]


def get_lollipop_graph():
    return dgl.remove_self_loop(MiniGCDataset(8, 7, 8)[3][0])


def get_hypercube_graph():
    return dgl.remove_self_loop(MiniGCDataset(8, 8, 9)[4][0])


def get_clique_graph():
    return dgl.remove_self_loop(MiniGCDataset(8, 6, 7)[6][0])
