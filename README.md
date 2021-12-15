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

![](https://raw.githubusercontent.com/arangoml/dgl-adapter/master/examples/assets/adb_logo.png)

![](https://raw.githubusercontent.com/arangoml/dgl-adapter/master/examples/assets/dgl_logo.png)

The ArangoDB-DGL Adapter exports Graphs from ArangoDB, a multi-model Graph Database, into Deep Graph Library (DGL), a python package for graph neural networks, and vice-versa.


## About DGL

Website: https://www.dgl.ai/

Documentation: https://docs.dgl.ai/

##  Quickstart

(TODO) Get Started on Colab: <a href="https://colab.research.google.com/github/arangoml/dgl-adapter/blob/master/examples/ArangoDB_DGL_Adapter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```py
from dgl.data import KarateClubDataset
from adbdgl_adapter.adbdgl_adapter import ArangoDB_DGL_Adapter

con = {
    "hostname": "localhost",
    "protocol": "http",
    "port": 8529,
    "username": "root",
    "password": "rootpassword",
    "dbName": "_system",
}

adbdgl_adapter = ArangoDB_DGL_Adapter(con)

# (Assume ArangoDB fraud-detection data dump is imported)
fraud_dgl_g = adbdgl_adapter.arangodb_graph_to_dgl("fraud-detection")
fraud_dgl_g_2 = adbdgl_adapter.arangodb_collections_to_dgl(
        "fraud-detection", 
        {"account", "Class", "customer"},
        {"accountHolder", "Relationship", "transaction"},
)


karate_dgl_g = KarateClubDataset()[0]
karate_adb_g = adbdgl_adapter.dgl_to_arangodb("Karate", karate_dgl_g)
```

##  Development & Testing

Prerequisite: `arangorestore` must be installed

1. `git clone https://github.com/arangoml/dgl-adapter.git`
2. `cd dgl-adapter`
3. `python -m venv .venv`
4. `source .venv/bin/activate` (MacOS) or `.venv/scripts/activate` (Windows)
5. `cd adbdgl_adapter`
6. `pip install -e . pytest`
7. `pytest`
    * If you encounter `ModuleNotFoundError`, try closing & relaunching your virtual environment by running `deactivate` in your terminal & restarting from Step 4.
