import os
import time
import json
import requests
import subprocess
from pathlib import Path

from dgl import remove_self_loop
from dgl.data import KarateClubDataset
from dgl.data import MiniGCDataset

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

    global db
    url = (
        conn.get("protocol", "https")
        + "://"
        + conn["hostname"]
        + ":"
        + str(conn["port"])
    )

    client = ArangoClient(hosts=url)
    db = client.db(conn["dbName"], conn["username"], conn["password"], verify=True)

    # for g in db.graphs():
    #     db.delete_graph(g['name'])

    # for col in db.collections():
    #     if col['system'] is False:
    #         db.delete_collection(col['name'])

    arango_restore("adbdgl_adapter/tests/data/fraud_dump")


def get_oasis_crendetials() -> dict:
    url = "https://tutorials.arangodb.cloud:8529/_db/_system/tutorialDB/tutorialDB"
    request = requests.post(url, data=json.dumps("{}"))
    if request.status_code != 200:
        raise Exception("Error retrieving login data.")

    return json.loads(request.text)


def arango_restore(path_to_data):
    restore_prefix = "./" if os.getenv("GITHUB_ACTIONS") else ""  # temporary hack

    subprocess.check_call(
        f'chmod -R 755 ./arangorestore && {restore_prefix}arangorestore -c none --server.endpoint http+ssl://{conn["hostname"]}:{conn["port"]} --server.username {conn["username"]} --server.database {conn["dbName"]} --server.password {conn["password"]} --include-system-collections true --input-directory "{PROJECT_DIR}/{path_to_data}"',
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
    return remove_self_loop(MiniGCDataset(8, 7, 8)[3][0])


def get_hypercube_graph():
    return remove_self_loop(MiniGCDataset(8, 8, 9)[4][0])


def get_clique_graph():
    return remove_self_loop(MiniGCDataset(8, 6, 7)[6][0])
