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

The ArangoDB-DGL Adapter exports Graphs from ArangoDB, the multi-model database for graph & beyond, into Deep Graph Library (DGL), a python package for graph neural networks, and vice-versa.


## About DGL

The Deep Graph Library (DGL) is an easy-to-use, high performance and scalable Python package for deep learning on graphs. DGL is framework agnostic, meaning if a deep graph model is a component of an end-to-end application, the rest of the logics can be implemented in any major frameworks, such as PyTorch, Apache MXNet or TensorFlow.

* [Website](https://www.dgl.ai/)
* [Documentation](https://docs.dgl.ai/)
* [Highlighted Features](https://github.com/dmlc/dgl#highlighted-features)

## Installation

#### Latest Release
```
pip install adbdgl-adapter
```
#### Current State
```
pip install git+https://github.com/arangoml/dgl-adapter.git
```

##  Quickstart

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arangoml/dgl-adapter/blob/master/examples/ArangoDB_DGL_Adapter.ipynb)

Also available as an ArangoDB Lunch & Learn session: [Graph & Beyond Course #2.8](https://www.arangodb.com/resources/lunch-sessions/graph-beyond-lunch-break-2-8-dgl-adapter/)

```py
import pandas
import torch
from dgl import heterograph

from arango import ArangoClient  # Python-Arango driver

from adbdgl_adapter import ADBDGL_Adapter, ADBDGL_Controller
from adbdgl_adapter.encoders import IdentityEncoder, CategoricalEncoder

# Let's assume that the ArangoDB "IMDB" dataset is imported to this endpoint
db = ArangoClient(hosts="http://localhost:8529").db("_system", username="root", password="")

hetero_graph = dgl.heterograph({
    ("user", "follows", "user"): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ("user", "follows", "topic"): (torch.tensor([1, 1]), torch.tensor([1, 2])),
    ("user", "plays", "game"): (torch.tensor([0, 3]), torch.tensor([3, 4])),
})
hetero_graph.nodes["user"].data["features"] = torch.tensor([21, 44, 16, 25])
hetero_graph.nodes["user"].data["label"] = torch.tensor([1, 2, 0, 1])
hetero_graph.nodes["game"].data["features"] = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
hetero_graph.edges[("user", "plays", "game")].data["features"] = torch.tensor([[6, 1], [1000, 0]])

adbdgl_adapter = ADBDGL_Adapter(db)
```

### DGL to ArangoDB
```py
# 1.1: DGL to ArangoDB
adb_g = adbdgl_adapter.dgl_to_arangodb("HeteroGraph", hetero_graph)

# 1.2: DGL to ArangoDB with a (completely optional) metagraph for customized adapter behaviour
def label_tensor_to_2_column_dataframe(dgl_tensor):
    """
    A user-defined function to create two
    ArangoDB attributes out of the 'user' label tensor

    NOTE: user-defined functions must return a Pandas Dataframe
    """
    label_map = {0: "Class A", 1: "Class B", 2: "Class C"}

    df = pandas.DataFrame(columns=["label_num", "label_str"])
    df["label_num"] = pyg_tensor.tolist()
    df["label_str"] = df["label_num"].map(label_map)

    return df


metagraph = {
    "nodeTypes": {
        "user": {
            "features": "user_age",  # 1) you can specify a string value for attribute renaming
            "label": label_tensor_to_2_column_dataframe,  # 2) you can specify a function for user-defined handling, as long as the function returns a Pandas DataFrame
        },
        "game": {
            # 3) you can specify a list of strings for tensor dissasembly (if you know the number of node/edge features in advance)
            "features": ["is_multiplayer", "is_free_to_play"]
        },
    },
    "edgeTypes": {
        ("user", "plays", "game"): {
            # 3) you can specify a list of strings for tensor dissasembly (if you know the number of node/edge features in advance)
            "features": ["hours_played", "is_satisfied_with_game"]
        },
    },
}


adb_g = adbdgl_adapter.dgl_to_arangodb("HeteroGraph", hetero_graph, metagraph, explicit_metagraph=False)

# 1.3: DGL to ArangoDB with the same (optional) metagraph, but with `explicit_metagraph=True`
# With `explicit_metagraph=True`, the node & edge types omitted from the metagraph will NOT be converted to ArangoDB.
# Only 'user', 'game', and ('user', 'plays', 'game') will be brought over (i.e 'topic', ('user', 'follows', 'user'), ... are ignored)
adb_g = adbdgl_adapter.dgl_to_arangodb("HeteroGraph", hetero_graph, metagraph, explicit_metagraph=True)

# 1.4: DGL to ArangoDB with a Custom Controller  (more user-defined behavior)
class Custom_ADBDGL_Controller(ADBDGL_Controller):
    def _prepare_dgl_node(self, dgl_node: dict, node_type: str) -> dict:
        """Optionally modify a DGL node object before it gets inserted into its designated ArangoDB collection.

        :param pyg_node: The DGL node object to (optionally) modify.
        :param node_type: The DGL Node Type of the node.
        :return: The DGL Node object
        """
        dgl_node["foo"] = "bar"
        return dgl_node

    def _prepare_dgl_edge(self, dgl_edge: dict, edge_type: tuple) -> dict:
        """Optionally modify a DGL edge object before it gets inserted into its designated ArangoDB collection.

        :param dgl_edge: The DGL edge object to (optionally) modify.
        :param edge_type: The Edge Type of the DGL edge. Formatted
            as (from_collection, edge_collection, to_collection)
        :return: The DGL Edge object
        """
        dgl_edge["bar"] = "foo"
        return dgl_edge


adb_g = ADBDGL_Adapter(db, Custom_ADBDGL_Controller()).dgl_to_arangodb("HeteroGraph", hetero_graph)
```

### ArangoDB to DGL
```py
# Start from scratch!
db.delete_graph("HeteroGraph", drop_collections=True, ignore_missing=True)
adbdgl_adapter.dgl_to_arangodb("HeteroGraph", hetero_graph)

# 2.1: ArangoDB to DGL via Graph name (does not transfer attributes)
dgl_g = adbdgl_adapter.arangodb_graph_to_dgl("HeteroGraph")

# 2.2: ArangoDB to DGL via Collection names (does not transfer attributes)
dgl_g = adbdgl_adapter.arangodb_collections_to_dgl("HeteroGraph", v_cols={"user", "game"}, e_cols={"plays"})

# 2.3: ArangoDB to DGL via Metagraph v1 (transfer attributes "as is", meaning they are already formatted to DGL data standards)
metagraph_v1 = {
    "vertexCollections": {
        # we instruct the adapter to create the "features" and "label" tensor data from the "features" and "label" ArangoDB attributes
        "user": {"features": "features", "label": "label"},
        "game": {"features": "features"},
        "topic": {},
    },
    "edgeCollections": {"plays": {"features": "features"}, "follows": {}},
}
dgl_g = adbdgl_adapter.arangodb_to_dgl("HeteroGraph", metagraph_v1)

# 2.4: ArangoDB to DGL via Metagraph v2 (transfer attributes via user-defined encoders)
# For more info on user-defined encoders, see https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
metagraph_v2 = {
    "vertexCollections": {
        "Movies": {
            "x": {  # Build a feature matrix from the "Action" & "Drama" document attributes
                "Action": IdentityEncoder(dtype=torch.long),
                "Drama": IdentityEncoder(dtype=torch.long),
            },
            "y": "Comedy",
        },
        "Users": {
            "x": {
                "Gender": CategoricalEncoder(mapping={"M": 0, "F": 1}),
                "Age": IdentityEncoder(dtype=torch.long),
            }
        },
    },
    "edgeCollections": {"Ratings": {"edge_weight": "Rating"}},
}
dgl_g = adbdgl_adapter.arangodb_to_dgl("IMDB", metagraph_v2)

# 2.5: ArangoDB to DGL via Metagraph v3 (transfer attributes via user-defined functions)
def udf_user_features(user_df):
    # process the user_df Pandas DataFrame to return a feature matrix in a tensor
    # user_df["features"] = ...
    return torch.tensor(user_df["features"].to_list())


def udf_game_features(game_df):
    # process the game_df Pandas DataFrame to return a feature matrix in a tensor
    # game_df["features"] = ...
    return torch.tensor(game_df["features"].to_list())


metagraph_v3 = {
    "vertexCollections": {
        "user": {
            "features": udf_user_features,  # supports named functions
            "label": lambda df: torch.tensor(
                df["label"].to_list()
            ),  # also supports lambda functions
        },
        "game": {"features": udf_game_features},
    },
    "edgeCollections": {
        "plays": {"features": (lambda df: torch.tensor(df["features"].to_list()))},
    },
}
dgl_g = adbdgl_adapter.arangodb_to_dgl("HeteroGraph", metagraph_v3)
```

##  Development & Testing

Prerequisite: `arangorestore`

1. `git clone https://github.com/arangoml/dgl-adapter.git`
2. `cd dgl-adapter`
3. (create virtual environment of choice)
4. `pip install -e .[dev]`
5. (create an ArangoDB instance with method of choice)
6. `pytest --url <> --dbName <> --username <> --password <>`

**Note**: A `pytest` parameter can be omitted if the endpoint is using its default value:
```python
def pytest_addoption(parser):
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="")
```
