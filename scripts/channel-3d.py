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

    s = radius * 0.7 / math.sqrt(2)
    r_coord = radius / math.sqrt(2)
    
    z_coords = [0, length]
 
    for z in z_coords:
        suffix = f"_z{int(z==length)}"
        bmd.add_vertex(-s, -s, z, f"v0{suffix}")
        bmd.add_vertex( s, -s, z, f"v1{suffix}")
        bmd.add_vertex( s,  s, z, f"v2{suffix}")
        bmd.add_vertex(-s,  s, z, f"v3{suffix}")
        bmd.add_vertex(-r_coord, -r_coord, z, f"v4{suffix}")
        bmd.add_vertex( r_coord, -r_coord, z, f"v5{suffix}")
        bmd.add_vertex( r_coord,  r_coord, z, f"v6{suffix}")
        bmd.add_vertex(-r_coord,  r_coord, z, f"v7{suffix}")

    bmd.add_hexblock(["v0_z0", "v1_z0", "v2_z0", "v3_z0", "v0_z1", "v1_z1", "v2_z1", "v3_z1"], 
                     (n_theta, n_theta, n_z), "mid", "main")
    bmd.add_hexblock(["v4_z0", "v5_z0", "v1_z0", "v0_z0", "v4_z1", "v5_z1", "v1_z1", "v0_z1"], 
                     (n_theta, n_r, n_z), "b_bottom", "main")
    bmd.add_hexblock(["v5_z0", "v6_z0", "v2_z0", "v1_z0", "v5_z1", "v6_z1", "v2_z1", "v1_z1"], 
                     (n_theta, n_r, n_z), "b_right", "main")
    bmd.add_hexblock(["v6_z0", "v7_z0", "v3_z0", "v2_z0", "v6_z1", "v7_z1", "v3_z1", "v2_z1"], 
                     (n_theta, n_r, n_z), "b_top", "main")
    bmd.add_hexblock(["v7_z0", "v4_z0", "v0_z0", "v3_z0", "v7_z1", "v4_z1", "v0_z1", "v3_z1"], 
                     (n_theta, n_r, n_z), "b_left", "main")
    for z in z_coords:
        suf = f"_z{int(z==length)}"
        bmd.add_arcedge(["v4"+suf, "v5"+suf], "arc_bot"+suf, Point(0, -radius, z))
        bmd.add_arcedge(["v5"+suf, "v6"+suf], "arc_rig"+suf, Point(radius, 0, z))
        bmd.add_arcedge(["v6"+suf, "v7"+suf], "arc_top"+suf, Point(0, radius, z))
        bmd.add_arcedge(["v7"+suf, "v4"+suf], "arc_lef"+suf, Point(-radius, 0, z))

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
    create_cylinder_mesh(radius=args.H[1]/2, length=args.H[0], n_r=args.n[1], n_theta=args.n[2], n_z=args.n[0], case_dir=args.case)