# ArangoDB-DGL Adapter

[![build](https://github.com/arangoml/dgl-adapter/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/arangoml/dgl-adapter/actions/workflows/build.yml)
[![CodeQL](https://github.com/arangoml/dgl-adapter/actions/workflows/analyze.yml/badge.svg?branch=master)](https://github.com/arangoml/dgl-adapter/actions/workflows/analyze.yml)
[![Coverage Status](https://coveralls.io/repos/github/arangoml/dgl-adapter/badge.svg?branch=master)](https://coveralls.io/github/arangoml/dgl-adapter)
[![Last commit](https://img.shields.io/github/last-commit/arangoml/dgl-adapter)](https://github.com/arangoml/dgl-adapter/commits/master)

[![PyPI version badge](https://img.shields.io/pypi/v/adbdgl-adapter?color=3775A9&style=for-the-badge&logo=pypi&logoColor=FFD43B)](https://pypi.org/project/adbdgl-adapter/)
[![Python versions badge](https://img.shields.io/pypi/pyversions/adbdgl-adapter?color=3776AB&style=for-the-badge&logo=python&logoColor=FFD43B)](https://pypi.org/project/adbdgl-adapter/)

[![License](https://img.shields.io/github/license/arangoml/dgl-adapter?color=9E2165&style=for-the-badge)](https://github.com/arangoml/dgl-adapter/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/static/v1?style=for-the-badge&label=code%20style&message=black&color=black)](https://github.com/psf/black)
[![Downloads](https://img.shields.io/badge/dynamic/json?style=for-the-badge&color=282661&label=Downloads&query=total_downloads&url=https://api.pepy.tech/api/projects/adbdgl-adapter)](https://pepy.tech/project/adbdgl-adapter)


<a href="https://www.arangodb.com/" rel="arangodb.com">![](https://raw.githubusercontent.com/arangoml/dgl-adapter/master/examples/assets/adb_logo.png)</a>
<a href="https://www.dgl.ai/" rel="dgl.ai"><img src="https://raw.githubusercontent.com/arangoml/dgl-adapter/master/examples/assets/dgl_logo.png" width=40% /></a>

The ArangoDB-DGL Adapter exports Graphs from ArangoDB, a multi-model Graph Database, into Deep Graph Library (DGL), a python package for graph neural networks, and vice-versa.


## About DGL

The Deep Graph Library (DGL) is an easy-to-use, high performance and scalable Python package for deep learning on graphs. DGL is framework agnostic, meaning if a deep graph model is a component of an end-to-end application, the rest of the logics can be implemented in any major frameworks, such as PyTorch, Apache MXNet or TensorFlow.

* [Website](https://www.dgl.ai/)
* [Documentation](https://docs.dgl.ai/)
* [Highlighted Features](https://github.com/dmlc/dgl#highlighted-features)

##  Quickstart

Get Started on Colab: <a href="https://colab.research.google.com/github/arangoml/dgl-adapter/blob/master/examples/ArangoDB_DGL_Adapter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```py
# Import the ArangoDB-DGL Adapter
from adbdgl_adapter.adapter import ADBDGL_Adapter

# Import a sample graph from DGL
from dgl.data import KarateClubDataset

# This is the connection information for your ArangoDB instance
# (Let's assume that the ArangoDB fraud-detection data dump is imported to this endpoint)
con = {
    "hostname": "localhost",
    "protocol": "http",
    "port": 8529,
    "username": "root",
    "password": "rootpassword",
    "dbName": "_system",
}

# This instantiates your ADBDGL Adapter with your connection credentials
adbdgl_adapter = ADBDGL_Adapter(con)

# ArangoDB to DGL via Graph
dgl_fraud_graph = adbdgl_adapter.arangodb_graph_to_dgl("fraud-detection")

# ArangoDB to DGL via Collections
dgl_fraud_graph_2 = adbdgl_adapter.arangodb_collections_to_dgl(
        "fraud-detection", 
        {"account", "Class", "customer"}, # Specify vertex collections
        {"accountHolder", "Relationship", "transaction"}, # Specify edge collections
)

# ArangoDB to DGL via Metagraph
metagraph = {
    "vertexCollections": {
        "account": {"Balance", "account_type", "customer_id", "rank"},
        "customer": {"Name", "rank"},
    },
    "edgeCollections": {
        "transaction": {"transaction_amt", "sender_bank_id", "receiver_bank_id"},
        "accountHolder": {},
    },
}
dgl_fraud_graph_3 = adbdgl_adapter.arangodb_to_dgl("fraud-detection", metagraph)

# DGL to ArangoDB
dgl_karate_graph = KarateClubDataset()[0]
adb_karate_graph = adbdgl_adapter.dgl_to_arangodb("Karate", karate_dgl_g)
```

##  Development & Testing

Prerequisite: `arangorestore`

1. `git clone https://github.com/arangoml/dgl-adapter.git`
2. `cd dgl-adapter`
3. (create virtual environment of choice)
5. `pip install -e . pytest`
6. `docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD=opensesame arangodb/arangodb:3.9.1` (optional if not already active)
7. `pytest --host localhost --port 8529 --password opensesame`

Note: You can run the set of tests on any ArangoDB endpoint via the following command-line options:
```python
def pytest_addoption(parser):
    parser.addoption("--protocol", action="store", default="http")
    parser.addoption("--host", action="store", default="localhost")
    parser.addoption("--port", action="store", default="8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="opensesame")
```