import argparse
import math
from pathlib import Path
import numpy as np
from ofblockmeshdicthelper import BlockMeshDict, Face

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def create_cylinder_mesh(radius, length, n_r, n_theta, n_z, case_dir):
    bmd = BlockMeshDict()
    bmd.set_metric("m")

    # 1. 计算坐标参数
    # 内部正方形边长一半 (s)
    s = radius * 0.7 / math.sqrt(2)
    # 外部圆周 45度角位置的坐标 (R')
    r_coord = radius / math.sqrt(2)
    
    z_coords = [0, length]
    
    # 2. 定义顶点
    for z in z_coords:
        suffix = f"_z{int(z==length)}"
        # 内部正方形 4 点: 0:左下, 1:右下, 2:右上, 3:左上
        bmd.add_vertex(-s, -s, z, f"v0{suffix}")
        bmd.add_vertex( s, -s, z, f"v1{suffix}")
        bmd.add_vertex( s,  s, z, f"v2{suffix}")
        bmd.add_vertex(-s,  s, z, f"v3{suffix}")
        
        # 外部圆周 4 点 (与内部点放射状对齐): 4:左下, 5:右下, 6:右上, 7:左上
        bmd.add_vertex(-r_coord, -r_coord, z, f"v4{suffix}")
        bmd.add_vertex( r_coord, -r_coord, z, f"v5{suffix}")
        bmd.add_vertex( r_coord,  r_coord, z, f"v6{suffix}")
        bmd.add_vertex(-r_coord,  r_coord, z, f"v7{suffix}")

    # 3. 定义 5 个 Hex 区块 (必须严格遵守顶点顺序)
    # 中间块: v0, v1, v2, v3
    bmd.add_hexblock(["v0_z0", "v1_z0", "v2_z0", "v3_z0", "v0_z1", "v1_z1", "v2_z1", "v3_z1"], 
                     (n_theta, n_theta, n_z), "mid", "main")
    
    # 下方块: v4, v5, v1, v0
    bmd.add_hexblock(["v4_z0", "v5_z0", "v1_z0", "v0_z0", "v4_z1", "v5_z1", "v1_z1", "v0_z1"], 
                     (n_theta, n_r, n_z), "b_bottom", "main")
    
    # 右侧块: v5, v6, v2, v1
    bmd.add_hexblock(["v5_z0", "v6_z0", "v2_z0", "v1_z0", "v5_z1", "v6_z1", "v2_z1", "v1_z1"], 
                     (n_theta, n_r, n_z), "b_right", "main")
    
    # 上方块: v6, v7, v3, v2
    bmd.add_hexblock(["v6_z0", "v7_z0", "v3_z0", "v2_z0", "v6_z1", "v7_z1", "v3_z1", "v2_z1"], 
                     (n_theta, n_r, n_z), "b_top", "main")
    
    # 左侧块: v7, v4, v0, v3
    bmd.add_hexblock(["v7_z0", "v4_z0", "v0_z0", "v3_z0", "v7_z1", "v4_z1", "v0_z1", "v3_z1"], 
                     (n_theta, n_r, n_z), "b_left", "main")

    # 4. 定义圆弧边 (取 0, 90, 180, 270 度作为中点)
    for z in z_coords:
        suf = f"_z{int(z==length)}"
        bmd.add_arcedge(["v4"+suf, "v5"+suf], "arc_bot"+suf, Point(0, -radius, z))
        bmd.add_arcedge(["v5"+suf, "v6"+suf], "arc_rig"+suf, Point(radius, 0, z))
        bmd.add_arcedge(["v6"+suf, "v7"+suf], "arc_top"+suf, Point(0, radius, z))
        bmd.add_arcedge(["v7"+suf, "v4"+suf], "arc_lef"+suf, Point(-radius, 0, z))

    # 5. 定义边界 (Faces)
    inlet_faces = [
        Face(("v0_z0", "v3_z0", "v2_z0", "v1_z0"), "in_mid"), 
        Face(("v4_z0", "v0_z0", "v1_z0", "v5_z0"), "in_bot"), 
        Face(("v5_z0", "v1_z0", "v2_z0", "v6_z0"), "in_rig"), 
        Face(("v6_z0", "v2_z0", "v3_z0", "v7_z0"), "in_top"), 
        Face(("v7_z0", "v3_z0", "v0_z0", "v4_z0"), "in_lef")  
    ]
    
    outlet_faces = [
        Face(("v0_z1", "v3_z1", "v2_z1", "v1_z1"), "out_mid"),
        Face(("v4_z1", "v0_z1", "v1_z1", "v5_z1"), "out_bot"),
        Face(("v5_z1", "v1_z1", "v2_z1", "v6_z1"), "out_rig"),
        Face(("v6_z1", "v2_z1", "v3_z1", "v7_z1"), "out_top"),
        Face(("v7_z1", "v3_z1", "v0_z1", "v4_z1"), "out_lef")
    ]
    
    wall_faces = [
        Face(("v4_z0", "v4_z1", "v5_z1", "v5_z0"), "w_bot"),
        Face(("v5_z0", "v5_z1", "v6_z1", "v6_z0"), "w_rig"),
        Face(("v6_z0", "v6_z1", "v7_z1", "v7_z0"), "w_top"),
        Face(("v7_z0", "v7_z1", "v4_z1", "v4_z0"), "w_lef")
    ]

    bmd.add_boundary("patch", "inlets", inlet_faces)
    bmd.add_boundary("patch", "outlets", outlet_faces)
    bmd.add_boundary("wall", "walls", wall_faces)

    # 分配 ID 并格式化
    bmd.assign_vertexid()
    bmd_path = Path(case_dir) / "system" / "blockMeshDict"
    bmd_path.parent.mkdir(exist_ok=True, parents=True)
    bmd_path.write_text(bmd.format())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", nargs=3, type=float)
    parser.add_argument("-n", nargs=3, type=int)
    parser.add_argument("case")
    args = parser.parse_args()
    # 映射参数: n[0]->轴向, n[1]->径向, n[2]->周向
    create_cylinder_mesh(radius=args.H[1]/2, length=args.H[0], n_r=args.n[1], n_theta=args.n[2], n_z=args.n[0], case_dir=args.case)