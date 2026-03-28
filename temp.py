import os
import random
from pathlib import Path

# ====== 配置路径 ======
src_dir = Path("/data1/turbdiff/shapes/data")
dst_root = Path("/data1/turbdiff/shapes")  # 当前目录下创建 shapes

# ====== 读取类别 ======
all_items = [p for p in src_dir.iterdir() if p.is_dir()]

assert len(all_items) == 30, f"Expected 30 items, got {len(all_items)}"

# ====== 随机打乱 ======
random.seed(42)  # 固定随机性（可改）
random.shuffle(all_items)

# ====== 划分 ======
train_items = all_items[:18]
val_items = all_items[18:24]
test_items = all_items[24:]

splits = {
    "train": train_items,
    "val": val_items,
    "test": test_items
}

# ====== 创建目录 + 软链接 ======
for split, items in splits.items():
    split_dir = dst_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        target = item.resolve()
        link_name = split_dir / item.name

        # 如果已存在就跳过（或删除重建）
        if link_name.exists():
            continue

        os.symlink(target, link_name)

print("Done!")
print("Train:", [p.name for p in train_items])
print("Val:", [p.name for p in val_items])
print("Test:", [p.name for p in test_items])