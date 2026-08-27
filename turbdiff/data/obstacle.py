import os
import torch
import trimesh
import diffusion_net
from torch.utils.data import Dataset


class SingleObstacleData(Dataset):
    def __init__(self, file_path, k_eig=64, op_cache_dir=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        self.file_path = file_path
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        path = self.file_path
        
        try:
            mesh = trimesh.load(path, force='mesh')
        except Exception as e:
            raise RuntimeError(f"Error loading {path}: {e}")

        mesh.merge_vertices() 
        mesh.remove_unreferenced_vertices()

        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()

        verts = diffusion_net.geometry.normalize_positions(verts)

        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            verts, faces, 
            k_eig=self.k_eig, 
            op_cache_dir=self.op_cache_dir
        )
        
        label = torch.tensor(0).long() 
        if torch.isnan(evals).any() or torch.isnan(L.to_dense()).any():
            print(f"!!! CRITICAL: Geometric operators contain NaN for mesh: {self.file_path}")

        return {
        "verts": verts,
        "faces": faces,
        "frames": frames,
        "mass": mass,
        "L": L,
        "evals": evals,
        "evecs": evecs,
        "gradX": gradX,
        "gradY": gradY
        }

