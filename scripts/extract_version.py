# -*- coding: utf-8 -*-
import requests

if __name__ == "__main__":
    response = requests.get(
        "https://api.github.com/repos/arangoml/dgl-adapter/releases/latest"
    )
    response.raise_for_status()
    print(response.json().get("tag_name", "0.0.0"))
