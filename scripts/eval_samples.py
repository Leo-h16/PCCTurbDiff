#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
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


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate existing samples"
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cuda"
    )

    parser.add_argument(
        "--expensive",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "ckpt",
        help="Path to .ckpt file"
    )

    parser.add_argument(
        "samples_path",
        help="Existing .h5 sample file"
    )

    parser.add_argument(
        "overrides",
        nargs="*"
    )


    args = parser.parse_args()


    device = torch.device(args.device)

    ckpt_path = args.ckpt
    samples_path = Path(args.samples_path)

    expensive_metrics = args.expensive


    # ==============================
    # load config from checkpoint
    # ==============================

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu"
    )


    if "config" not in ckpt:
        log.error("Checkpoint has no config")
        sys.exit(1)


    run_config = OmegaConf.create(
        ckpt["config"]
    )

    config = OmegaConf.merge(
        run_config,
        OmegaConf.from_cli(args.overrides)
    )


    print_config(config)



    # ==============================
    # create datamodule/task
    # ==============================

    datamodule, task = instantiate_data_and_task(
        config
    )

    task.load_state_dict(
        ckpt["state_dict"]
    )

    task = task.to(device)
    task.eval()


    datamodule.setup("test")


    stats = move_data_to_device(
        datamodule.stats,
        device
    )


    # ==============================
    # load existing samples
    # ==============================

    sample_store = SampleStore(
        samples_path,
        task.variables
    )


    print(
        "Available cases:",
        len(sample_store.case_names)
    )


    # ==============================
    # create metrics
    # ==============================

    data_root = (
        Path(config.data.root)
        / "test"
    )


    metrics = task._sample_metrics(
        "test",
        data_root
    ).to(device)


    # ==============================
    # compute metrics
    # ==============================

    print("Start computing metrics")


    log_metrics = metrics.compute(
        sample_store,
        stats,
        device,
        expensive_metrics=expensive_metrics
    )


    print("Metrics computed")


    log_metrics = {
        k: float(v.item())
        for k, v in log_metrics.items()
    }


    for key in sorted(log_metrics.keys()):
        print(
            f"{key}: {log_metrics[key]}"
        )


if __name__ == "__main__":
    main()