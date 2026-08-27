import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
import shutil
from typing import List


@dataclass(frozen=True)
class StlObstacle:
    path: Path
    offset: float
    name: str


@dataclass(frozen=True)
class ChannelConfig:
    h: tuple[float, float, float] = (0.4, 0.1, 0.1)
    n: tuple[int, int, int] = (192, 48, 48)
    inflow: float = 20.0
    holes: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)
    two_dimensional: bool = False
    parallel: int = 1
    delta_t: float = 1e-5
    end_time: float = 1.0
    write_interval: float = 1e-3
    stl_obstacles: List[StlObstacle] = field(default_factory=list)
    

    def add_stl_obstacle(self, stl_path: str, offset: float = 0.0) -> "ChannelConfig":
        path_obj = Path(stl_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")

        new_obstacle = StlObstacle(
            path=path_obj.resolve(),
            offset=offset,
            name=path_obj.stem,
        )
        return replace(self, stl_obstacles=self.stl_obstacles + [new_obstacle])

    def add_basic_step(self, *, height: int, width: int, offset: int):
        hole = (offset, 0, 0, width, self.n[1], height)
        return replace(self, holes=self.holes + [hole])

    def add_top_step(self, *, height: int, width: int, offset: int):
        hole = (offset, 0, self.n[2] - height, width, self.n[1], height)
        return replace(self, holes=self.holes + [hole])

    def add_hole(self, *, x: int, y: int, z: int, width: int, depth: int, height: int):
        hole = (x, y, z, width, depth, height)
        return replace(self, holes=self.holes + [hole])

    def to_2d(self):
        hx, hy, hz = self.h
        nx, ny, nz = self.n
        return replace(
            self,
            h=(hx, hy / ny, hz),
            n=(nx, 1, nz),
            holes=[(x, 0, z, w, 1, h) for x, y, z, w, d, h in self.holes],
            two_dimensional=True,
        )
    
    def refine(self, scale: float):
        def scale_int(n: int) -> int:
            return round(n * scale)

        return replace(
            self,
            n=tuple(list(map(scale_int, self.n))),
            holes=[tuple(list(map(scale_int, hole))) for hole in self.holes],
        )
    
    @property
    def mid_y(self):
        return self.h[1] / 2.0

    @property
    def mid_z(self):
        return self.h[2] / 2.0


# ----------------------------
# snappyHexMesh
# ----------------------------

def setup_stl_mesh(case_dir: Path, config: ChannelConfig):
    if not config.stl_obstacles:
        return

    dx = config.h[0] / config.n[0]
    
    loc_x = 0.0  
    loc_y = 0.0  
    loc_z = config.h[0] * 0.75  

    tri_surface_dir = case_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    geometry_str = ""
    refinement_str = ""
    features_list_str = ""
    stl_names = []
    move_cmds = []

    for obs in config.stl_obstacles:
        dest_stl = tri_surface_dir / f"{obs.name}.stl"
        shutil.copy(obs.path, dest_stl)
        stl_names.append(f'"{obs.name}.stl"')

        target_z = obs.offset * dx
        scale_val = getattr(obs, 'scale', 1.0)

        transform_cmd = f"scale=({scale_val} {scale_val} {scale_val}), translate=(0 0 {target_z})"
        
        move_cmds.append(
            f'surfaceTransformPoints "{transform_cmd}" '
            f'constant/triSurface/{obs.name}.stl '
            f'constant/triSurface/{obs.name}.stl'
        )
        
        geometry_str += f"""
    {obs.name}.stl
    {{
        type triSurfaceMesh;
        name {obs.name};
    }}
"""
        refinement_str += f"""
        {obs.name}
        {{
            level (2 2);
            patchInfo
            {{
                type wall;
            }}
        }}
"""

    # --- surfaceFeaturesDict (OF10) ---
    stl_list_str = " ".join(stl_names)
    sfe_content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}

surfaces ({stl_list_str});

includedAngle 150;
"""
    (system_dir / "surfaceFeaturesDict").write_text(sfe_content)
    
    (case_dir / "pre-mesh.sh").write_text("#!/bin/sh\n" + "\n".join(move_cmds))

    # --- snappyHexMeshDict  ---
    shm_content = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
{geometry_str}
}}

castellatedMeshControls
{{
    maxLocalCells 1000000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features 
    (
{features_list_str}
    );

    refinementSurfaces
    {{
{refinement_str}
    }}

    resolveFeatureAngle 30;

    locationInMesh ({loc_x} {loc_y} {loc_z});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;

    nSmoothScale 4;
    errorReduction 0.75;
}}

mergeTolerance 1e-6;
"""
    (system_dir / "snappyHexMeshDict").write_text(shm_content)

# ----------------------------
# case
# ----------------------------

def generate_case(path: Path, config: ChannelConfig):
    les_cmd = [
        "python", "scripts/les-case.py",
        "--inflow", "0", "0", str(config.inflow),
        "--end-time", str(config.end_time),
        "--delta-t", str(config.delta_t),
        "--write-interval", str(config.write_interval),
        "--parallel", str(config.parallel),
        str(path),
    ]
    subprocess.run(les_cmd, check=True)

    setup_stl_mesh(path, config)

    channel_cmd = [
        "python", "scripts/channel-3d.py",
        "-H", *map(str, config.h),
        "-n", *map(str, config.n),
    ]
    for hole in config.holes:
        channel_cmd.append("--hole")
        channel_cmd.extend(map(str, hole))
    if config.two_dimensional:
        channel_cmd.append("--2d")
    channel_cmd.append(str(path))
    subprocess.run(channel_cmd, check=True)


    control_dict_path = path / "system" / "controlDict"

    collated_settings = f"""
// --- Collated File Handler Settings ---
optimisationSwitches
{{
    fileHandler     collated;
    maxProcessors   {config.parallel};
    maxThreadFileBufferSize 2e9; 
}}
"""
    if control_dict_path.exists():
        with open(control_dict_path, "a") as f:
            f.write(collated_settings)
        print(f"Successfully enabled collated mode for {config.parallel} cores.")
    else:
        print(f"Warning: {control_dict_path} not found, could not set collated mode.")