#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

from generate_utils import ChannelConfig, generate_case

def main():
    parser = argparse.ArgumentParser(description="Generate a single OpenFOAM case from an STL file.")
    parser.add_argument("root", help="Directory to generate the case into")
    parser.add_argument(
        "--stl-path", 
        required=True, 
        help="Path to the input .stl file"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=30,
        help="How far away to place the shape from the inlet",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Scale the resolution of the simulation by this factor",
    )
    
    args = parser.parse_args()
    
    offset = args.offset
    scale = args.scale
    stl_path = Path(args.stl_path)
    root = Path(args.root)

    if not stl_path.exists():
        print(f"Error: STL file '{stl_path}' not found.")
        sys.exit(1)

    config = ChannelConfig(
        h=(0.4, 0.1, 0.1),  
        n=(200, 15, 25),    
        inflow=20.0, 
        end_time=0.5, 
        write_interval=1e-4, 
        parallel=64
    )

  
    print(f"Processing STL: {stl_path.name} with offset {offset}")
    config = config.add_stl_obstacle(stl_path=str(stl_path.resolve()), offset=offset)

    if scale is not None:
        config = config.refine(scale)

    case_name = stl_path.stem  
    case_root = root / "data" / case_name / "case"
    
    print(f"Generating case into: {case_root}")
    generate_case(case_root, config)

    metadata = {
        "name": case_name,
        "source_stl": str(stl_path.resolve()),
        "offset": offset,
        "scale": scale
    }
    (case_root / "shape.json").write_text(json.dumps(metadata, indent=2))
    
    print("Done.")

if __name__ == "__main__":
    main()