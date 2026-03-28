import h5py
import numpy as np
import matplotlib
# 如果在服务器上运行，没有显示器，请取消下面这行的注释
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm

def verify_resampled_data(h5_path, time_idx=-1):
    print(f"正在尝试打开文件: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            # 1. 读取网格尺寸
            if "grid/cell_counts" not in f:
                print("错误: 找不到 grid/cell_counts")
                return
            
            nx, ny, nz = f["grid/cell_counts"][:]
            n_total = nx * ny * nz
            print(f"大网格尺寸: {nx} x {ny} x {nz} (总容量: {n_total})")

            cell_idx = f["grid/cell_idx"][:]
            print(f"物理核心点数: {len(cell_idx)}")

            # 2. 还原 3D Cell Type 网格
            tags_flat = np.full(n_total, 0, dtype=np.int32) 
            tags_flat[cell_idx] = f["grid/cell_type"][:]
            tags_3d = tags_flat.reshape(nx, ny, nz)

            # 3. 还原 3D 物理场网格
            u_mag = np.zeros((nx, ny, nz))
            if "data/u" in f:
                u_core = f["data/u"][time_idx]
                u_mag_core = np.linalg.norm(u_core, axis=-1)
                u_mag_flat = np.zeros(n_total)
                u_mag_flat[cell_idx] = u_mag_core
                u_mag = u_mag_flat.reshape(nx, ny, nz)

            # 4. 定义调色板
            colors = ['cyan', 'black', 'blue', 'yellow', 'red']
            cmap_tags = ListedColormap(colors)
            norm_tags = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap_tags.N)

            type_mapping = {"0": "Fluid", "1": "BG/Padding", "2": "Inlet", "3": "Outlet", "4": "Sphere"}

            # 5. 绘图
            print("正在生成图像...")
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 3)

            mid_x = nx // 2
            z_slices = [1, 13, nz - 2]
            
            for i, z_idx in enumerate(z_slices):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(tags_3d[:, :, z_idx].T, origin='lower', cmap=cmap_tags, norm=norm_tags, aspect='equal')
                ax.set_title(f"Z-Slice {z_idx}")

            ax_yz = fig.add_subplot(gs[1, 0:2])
            ax_yz.imshow(tags_3d[mid_x, :, :], origin='lower', cmap=cmap_tags, norm=norm_tags, aspect=1.0)
            ax_yz.set_title("Longitudinal View (Corrected)")

            # 保存并尝试显示
            output_file = "debug_check_v2.png"
            plt.savefig(output_file)
            print(f"!!! 图像已成功保存至: {output_file} !!!")
            
            # 如果你在本地运行，会弹出窗口；如果在服务器运行，此行可能会挂起
            print("尝试弹出窗口（如果没有弹出，请直接查看生成的 PNG 文件）...")
            plt.show()

    except Exception as e:
        print(f"运行中发生异常: {e}")

if __name__ == "__main__":
    print("脚本已启动...")
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="HDF5 file path or directory")
    args = parser.parse_args()
    
    path = Path(args.data).resolve()
    h5_file = path / "data.h5" if path.is_dir() else path
    
    if h5_file.exists():
        verify_resampled_data(h5_file)
    else:
        print(f"!!! 错误: 找不到文件 {h5_file} !!!")