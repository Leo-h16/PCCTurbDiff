#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

# 引入修改后的 generate_utils
from generate_utils import ChannelConfig, generate_case

def main():
    # 1. 解析参数
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

    # 2. 检查文件是否存在
    if not stl_path.exists():
        print(f"Error: STL file '{stl_path}' not found.")
        sys.exit(1)

    # 3. 准备配置
    # 基础配置
    config = ChannelConfig(
        h=(0.4, 0.1, 0.1),  # 长度 0.5m
        n=(200, 15, 25),    # 轴向200层，径向15层，周向30层
        inflow=20.0, 
        end_time=0.5, 
        write_interval=1e-4, 
        parallel=64
    )

    # 使用我们刚刚在 generate_utils 中添加的方法
    print(f"Processing STL: {stl_path.name} with offset {offset}")
    config = config.add_stl_obstacle(stl_path=str(stl_path.resolve()), offset=offset)

    if scale is not None:
        config = config.refine(scale)

    # 4. 生成 Case·
    case_name = stl_path.stem  # 使用文件名作为case名
    case_root = root / "data" / case_name / "case"
    
    print(f"Generating case into: {case_root}")
    generate_case(case_root, config)

    # 5. 写入元数据 shape.json
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