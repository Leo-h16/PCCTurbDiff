#!/usr/bin/env python
import argparse
import time
from pathlib import Path
import logging
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm

from turbdiff.config import instantiate_data_and_task
from turbdiff.data.ofles import Variable as V
from turbdiff.models.diffusion import DiffusionTraining
from turbdiff.models.regression import RegressionTraining

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode")
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("checkpoint_path", help="Path to checkpoint (.ckpt)")
    parser.add_argument("times_path", help="Directory to save times")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    times_dir = Path(args.times_path)
    times_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" in checkpoint:
        log.info("Load config from checkpoint")
        run_config = OmegaConf.create(checkpoint["config"])
    else:
        log.error("Checkpoint has no config")
        sys.exit(1)

    config = OmegaConf.merge(run_config)
    datamodule, task = instantiate_data_and_task(config)
    task.load_state_dict(checkpoint["state_dict"])
    task = task.to(device)

    datamodule.setup("test")
    dataset = datamodule.test_dataset

    torch.set_float32_matmul_precision("medium")

    times = []
    task.eval()
    with torch.no_grad():
        bar = tqdm(dataset.sample_idxs_by_file(), desc="Cases")
        for sample_idxs in bar:
            batch = dataset[[sample_idxs[0]]]
            batch = move_data_to_device(batch, device)

            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter_ns()
            
            if isinstance(task, DiffusionTraining):
                samples = task.sample(batch)
            elif isinstance(task, RegressionTraining):
                batch.data.t = batch.data.t[:, : task.context_window]
                batch.data.samples = {
                    v: sample[:, : task.context_window]
                    for v, sample in batch.data.samples.items()
                }
                u = batch.data.samples[V.U]
                batch.data.samples[V.U] = u + 0.01 * torch.randn_like(u)
                if args.mode == "init":
                    samples = task.unroll_samples(batch, [199], block_size=25)
                else:
                    samples = task.unroll_samples(batch, [21], block_size=25)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.perf_counter_ns()
            diff = end - start
            times.append(diff)
            bar.set_postfix({"time": diff / 10**9})

    times = np.array(times).astype(float) / 1e9
    file_name = f"{checkpoint_path.stem}.txt" if args.mode is None else f"{checkpoint_path.stem}-init.txt"
    np.savetxt(times_dir / file_name, times)

    print(times)
    print("min time:", times.min())

if __name__ == "__main__":
    main()

