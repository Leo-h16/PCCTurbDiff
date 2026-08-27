#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import shutil
from pathlib import Path
from turbdiff.openfoam import edit_openfoam_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inflow",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 10.0],
        help="Inflow velocity",
    )
    parser.add_argument("--end-time", type=float, default=0.1, help="End time")
    parser.add_argument("--delta-t", type=float, default=1e-5, help="Initial time step")
    parser.add_argument(
        "--write-interval", type=float, default=0.001, help="Write interval"
    )
    parser.add_argument(
        "-p", "--parallel", type=int, default=1, help="Number of parallel processes"
    )
    parser.add_argument("case", help="Path to generate the case at")
    args = parser.parse_args()

    inflow_velocity = args.inflow
    end_time = args.end_time
    delta_t = args.delta_t
    write_interval = args.write_interval
    parallel = args.parallel
    
    case_dir = Path(args.case)
    template_dir = Path(__file__).parent / "les-template"
    
    new_patch_name = case_dir.parent.name
    print(f"Detected STL patch name: {new_patch_name}")

    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(template_dir, case_dir)

    with edit_openfoam_dict(case_dir / "initial-conditions" / "U") as config:
        bf = config.assignments["boundaryField"]
        bf[new_patch_name] = { "type": "noSlip" }
        bf["inlets"]["value"].value = inflow_velocity

    with edit_openfoam_dict(case_dir / "initial-conditions" / "p") as config:
        config.assignments["boundaryField"][new_patch_name] = { "type": "zeroGradient" }

    with edit_openfoam_dict(case_dir / "initial-conditions" / "k") as config:
        config.assignments["boundaryField"][new_patch_name] = {
            "type": "kLowReWallFunction",
            "value": "uniform 0.01"
        }

    with edit_openfoam_dict(case_dir / "initial-conditions" / "nut") as config:
        config.assignments["boundaryField"][new_patch_name] = {
            "type": "nutkWallFunction",
            "value": "uniform 0"
        }
    with edit_openfoam_dict(case_dir / "system" / "controlDict") as config:
        config.assignments["endTime"] = end_time
        config.assignments["deltaT"] = delta_t
        config.assignments["writeInterval"] = write_interval

    with edit_openfoam_dict(case_dir / "system" / "decomposeParDict") as config:
        config.assignments["numberOfSubdomains"] = max(parallel, 1)

    initial_conditions_dir = case_dir / "initial-conditions"
    zero_dir = case_dir / "0.00000"
    if zero_dir.exists():
        shutil.rmtree(zero_dir)
    shutil.copytree(initial_conditions_dir, zero_dir)

if __name__ == "__main__":
    main()