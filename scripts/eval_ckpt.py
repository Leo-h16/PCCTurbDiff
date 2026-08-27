#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from turbdiff.config import instantiate_data_and_task
from turbdiff.models.metrics import SampleStore
from turbdiff.utils.logging import print_config
from turbdiff.utils.seed import manual_seed

log = logging.getLogger(__name__)


def save_metrics(results_dir, seed, ckpt_path, log_metrics):
    """Save one evaluation and refresh summaries of all evaluations in the dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "seed": str(seed),
        "checkpoint": str(Path(ckpt_path).resolve()),
        **log_metrics,
    }
    result_path = results_dir / f"seed_{seed}.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    records = [
        json.loads(path.read_text())
        for path in sorted(results_dir.glob("seed_*.json"))
    ]
    (results_dir / "summary.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    fieldnames = ["seed", "checkpoint"] + sorted(
        {
            key
            for record in records
            for key in record
            if key not in {"seed", "checkpoint"}
        }
    )
    with (results_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return result_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with overrides")
    parser.add_argument("--expensive", action="store_true", default=False)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-s", "--seed", default=2883413570083077179, type=int)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory for per-seed metrics and combined summary files",
    )
    parser.add_argument("ckpt", help="Path to .ckpt file")
    parser.add_argument("samples_path", help=".h5 file for storing samples")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    device = torch.device(args.device)
    seed = args.seed
    ckpt_path = args.ckpt
    samples_path = Path(args.samples_path)
    expensive_metrics = args.expensive
    overrides = args.overrides

    assert samples_path.suffix == ".h5"
    assert not samples_path.exists()

    ckpt = torch.load(ckpt_path) if ckpt_path is not None else {}
    if "config" in ckpt:
        log.info("Load config from checkpoint")
        run_config = OmegaConf.create(ckpt["config"])
    else:
        log.error("Checkpoint has no config")
        sys.exit(1)

    config = OmegaConf.merge(run_config, OmegaConf.from_cli(overrides))

    print_config(config)

    manual_seed(seed)
    torch.set_float32_matmul_precision(config.matmul_precision)

    datamodule, task = instantiate_data_and_task(config)
    task.load_state_dict(ckpt["state_dict"])
    task = task.to(device)
    datamodule.setup("test")

    sample_store = SampleStore(samples_path, task.variables)
    task.eval()
    with torch.no_grad():
        for batch in tqdm(datamodule.test_dataloader(), desc="Cases"):
            batch = move_data_to_device(batch, device)

            x_sample = task.sample(batch)

            sample_store.add_samples(x_sample, batch.data.metadata)
    data_root = Path(config.data.root) / "test" 
    metrics = task._sample_metrics("test", data_root).to(device)
    stats = move_data_to_device(datamodule.stats, device)
    log_metrics = metrics.compute(sample_store, stats, device, expensive_metrics=expensive_metrics)
    log_metrics = {key: float(value.item()) for key, value in log_metrics.items()}
    for key in sorted(log_metrics.keys()):
        value = log_metrics[key]
        print(f"{key}: {value}")

    if args.results_dir is not None:
        result_path = save_metrics(args.results_dir, seed, ckpt_path, log_metrics)
        print(f"Metrics saved to {result_path}")


if __name__ == "__main__":
    main()
