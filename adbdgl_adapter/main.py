import dgl
from dgl.heterograph import DGLHeteroGraph
import torch


g: DGLHeteroGraph = dgl.heterograph(
    {
        ("player", "plays", "game"): (
            torch.tensor([0, 1, 1, 2]),
            torch.tensor([0, 0, 1, 1]),
        ),
        ("developer", "develops", "game"): (torch.tensor([0, 0]), torch.tensor([0, 1])),
    }
)

ndata = {
    "h": {"player": None},
    "g": {"game": torch.ones(2, 1), "player": torch.tensor([1, 2, 3])},
}

for key, val in ndata.items():
    g.ndata[key] = val

print(g)
print(g.ndata["h"]["player"])
