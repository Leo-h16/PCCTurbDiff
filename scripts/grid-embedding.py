#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import h5py as h5
import numpy as np
from scipy.spatial import KDTree 
from tqdm import tqdm
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="OpenFOAM data directory")
    args = parser.parse_args()
    data_dir = Path(args.data)

    bbox = np.array([0.1, 0.1, 0.4]) 
    unpadded_counts = np.array([48, 48, 192]) 
    cell_counts = unpadded_counts + 2 
    n_total_geometry = np.prod(unpadded_counts)
    n_total_grid = np.prod(cell_counts)

    with h5.File(data_dir / "data.h5", mode="r+") as f:
        if "tmp" in f: del f["tmp"]
        f.move("data", "tmp")
        c_pos = f["domain/cell_centres"][:]
        u_ds, p_ds, k_ds, n_ds = f["tmp/u"], f["tmp/p"], f["tmp/k"], f["tmp/nut"]
        times = f['tmp/times']
        n_times = u_ds.shape[0]
        p_min, p_max = c_pos.min(axis=0), c_pos.max(axis=0)

        xi = np.linspace(p_min[0], p_max[0], unpadded_counts[0])
        yi = np.linspace(p_min[1], p_max[1], unpadded_counts[1])
        zi = np.linspace(p_min[2], p_max[2], unpadded_counts[2])
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
        grid_points = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]

        print("KDTree: Mapping refined mesh to uniform grid...")
        tree = KDTree(c_pos)
        d, tree_indices = tree.query(grid_points)
        local_coords = np.stack(np.meshgrid(
            np.arange(unpadded_counts[0]),
            np.arange(unpadded_counts[1]),
            np.arange(unpadded_counts[2]),
            indexing='ij'
        ), axis=-1).reshape(-1, 3)
        padded_coords = local_coords + 1
        cell_idx = np.ravel_multi_index(
            (padded_coords[:, 0], padded_coords[:, 1], padded_coords[:, 2]), 
            dims=tuple(cell_counts)
        )

        rad_sq = (gx.ravel() / (bbox[0]*0.5))**2 + (gy.ravel() / (bbox[1]*0.5))**2
        mask_wall = (rad_sq <= 1.0).astype(np.float32)

        dx = (p_max - p_min) / cell_counts
        threshold = np.linalg.norm(dx) * 0.7
        mask_geo = ((d>=threshold) & (mask_wall > 0)).astype(np.float32)

        z_min_val = gz.ravel().min()
        mask_inlet = ((gz.ravel() <= z_min_val + 1e-6)& (mask_wall > 0)).astype(np.float32)

        z_max_val = gz.ravel().max()
        mask_outlet = ((gz.ravel() >= z_max_val - 1e-6) & (mask_wall > 0)).astype(np.float32)

        mask_flow = mask_wall * (mask_geo == 0)

        condlist = [
            mask_geo == 1,
            mask_outlet == 1, 
            mask_inlet == 1,   
            mask_wall == 0  
        ]
        choicelist = [4, 3, 2, 1]
        
        cell_type = np.select(condlist, choicelist, default=0)


        if "geometry" in f: del f["geometry"]
        geo = f.create_group("geometry")
        geo["bounding_box"], geo["cell_counts"] = bbox, unpadded_counts

        if "grid" in f: del f["grid"]
        grid = f.create_group("grid")
        grid["cell_counts"], grid["cell_idx"], grid["cell_type"] = cell_counts, cell_idx, cell_type
        
        data= f.create_group("data")
        data["times"] = times
        ds_u = data.create_dataset("u", (n_times, n_total_geometry, 3), dtype='f4')
        ds_p = data.create_dataset("p", (n_times, n_total_geometry), dtype='f4')
        ds_k = data.create_dataset("k", (n_times, n_total_geometry), dtype='f4')
        ds_n = data.create_dataset("nut", (n_times, n_total_geometry), dtype='f4')

        for t in tqdm(range(n_times), desc="Voxelizing to 1D"):
            ds_u[t] = u_ds[t][tree_indices] * mask_flow[:, None]
            ds_p[t] = p_ds[t][tree_indices] * mask_flow
            ds_k[t] = k_ds[t][tree_indices] * mask_flow
            ds_n[t] = n_ds[t][tree_indices] * mask_flow

        del f["tmp"]
            


if __name__ == "__main__":
    main()