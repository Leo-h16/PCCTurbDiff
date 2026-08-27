#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm
import random
from turbdiff.config import instantiate_data_and_task
from turbdiff.data.ofles import Variable as V
from turbdiff.models.metrics import SampleStore
from turbdiff.utils.wandb import load_checkpoint, wandb_run
from turbdiff.utils.seed import manual_seed

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-n", "--samples", type=int, default=16, help="Number of samples per case")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("path", help="W&B run path")
    parser.add_argument("samples_path", help="Path for sample .h5 files")
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    device = torch.device(args.device)
    n_samples = args.samples
    batch_size = args.batch_size
    run_path = args.path
    samples_path = Path(args.samples_path)
    data_dir = Path(args.data_dir)
    overrides = args.overrides

    assert n_samples % batch_size == 0
    assert data_dir.is_dir()

    # Load run and config
    checkpoint = torch.load(run_path) if run_path is not None else {}
    if "config" in checkpoint:
        log.info("Load config from checkpoint")
        run_config = OmegaConf.create(checkpoint["config"])
    else:
        log.error("Checkpoint has no config")
        sys.exit(1)
    config = OmegaConf.merge(run_config, OmegaConf.from_cli(overrides))
    datamodule, task = instantiate_data_and_task(config)
    task.load_state_dict(checkpoint["state_dict"])
    task = task.to(device)


    manual_seed(817598073042842)  
    torch.set_float32_matmul_precision("high") 

    datamodule.setup("test")
    dataset = datamodule.test_dataset

    sample_store = SampleStore(samples_path, task.variables)

    if samples_path.exists():
        print("Samples already exist, moving directly to evaluation")
    else:
        task.eval()
        with torch.no_grad():
            rollout_steps = [22]  # DilResNet-22 rollout
            for sample_idxs in tqdm(dataset.sample_idxs_by_file(), desc="Cases"):
                if len(sample_idxs) <= rollout_steps[0]:
                    continue 
                    
                safe_sample_idxs = sample_idxs[:-rollout_steps[0]]  
                for _ in tqdm(range(n_samples // batch_size), desc="Batches", position=1):
                    selected_idxs = random.choices(safe_sample_idxs, k=batch_size)
                    batch = dataset[selected_idxs]

                    batch.data.t = batch.data.t[:, : task.context_window]
                    batch.data.samples = {
                        v: sample[:, : task.context_window]
                        for v, sample in batch.data.samples.items()
                    }
                    batch = move_data_to_device(batch, device)

                    u = batch.data.samples[V.U]
                    x_hat = task.unroll_samples(batch, rollout_steps, block_size=25)

                    sample_store.add_samples(x_hat[:, 0], batch.data.metadata)

    metrics = task._sample_metrics("test/rollout", data_dir).to(device)
    stats = move_data_to_device(datamodule.stats, device)
    log_metrics = metrics.compute(sample_store, stats, device)
    log_metrics = {key: float(value.item()) for key, value in log_metrics.items()}

    metrics_path = samples_path.parent / f"{Path(run_path).stem}-metrics.json"
    metrics_path.write_text(json.dumps(log_metrics, indent=4))

    for key in sorted(log_metrics.keys()):
        print(f"{key}: {log_metrics[key]}")


if __name__ == "__main__":
    main()