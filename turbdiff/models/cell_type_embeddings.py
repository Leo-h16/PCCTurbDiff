from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.ofles import OpenFOAMData

class CellTypeEmbedding(nn.Module):

    @staticmethod
    def create(type: Literal["learned", "onehot"], dim: int):
        if type == "learned":
            return CellTypeLearnedEmbedding(dim)
        elif type == "onehot":
            return CellTypeOneHotEmbedding()
        else:
            raise RuntimeError(f"Unknown cell type embedding {type}")

    def __init__(self):
        super().__init__()
        self.boundary_types = {
            "inside": 0,
            "walls": 1,
            "inlets": 2,
            "outlets": 3,
            "obstacle": 4,
        }

    @property
    def n_types(self):
        return len(self.boundary_types)

    @property
    def out_dim(self):
        raise NotImplementedError()

    def get_3d_cell_types(self, data: OpenFOAMData) -> torch.Tensor:
        grid_size = int(data.cell_counts.prod().item())

        cell_idx = data.cell_idx.reshape(-1)
        cell_type = data.cell_type.reshape(-1)

        grid_type = torch.full(
            (grid_size,),
            1,
            device=data.device,
            dtype=cell_type.dtype,
        )

        grid_type[cell_idx] = cell_type

        return grid_type.reshape(tuple(data.cell_counts))


class CellTypeLearnedEmbedding(CellTypeEmbedding):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(self.n_types, embedding_dim=dim)

    def forward(self, data: OpenFOAMData):
        idx_3d = self.get_3d_cell_types(data)
        embedded = self.embedding(idx_3d.long())
        return torch.movedim(embedded, -1, 0)

    @property
    def out_dim(self):
        return self.dim


class CellTypeOneHotEmbedding(CellTypeEmbedding):
    def forward(self, data: OpenFOAMData):
        idx_3d = self.get_3d_cell_types(data)
        one_hot = F.one_hot(idx_3d.long(), num_classes=self.n_types)
        return torch.movedim(one_hot, -1, 0).float()

    @property
    def out_dim(self):
        return self.n_types