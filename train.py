#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import faulthandler
import csv
import json
import logging
import math
import os
import warnings
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.connectors.signal_connector import _SignalConnector

from turbdiff.callbacks import ConfigInCheckpoint
from turbdiff.config import instantiate_data_and_task
from turbdiff.plots import OpenFOAMPlots
from turbdiff.time_limit import TimeLimit
from turbdiff.utils import (
    WandbModelCheckpoint,
    WandbSummaries,
    filter_device_available,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)


# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# Stop lightning from pestering us about things we already know
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)
warnings.filterwarnings(
    "ignore",
    "The dataloader, [^,]+, does not have many workers",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    filter_device_available
)


def if_eq(a, b, then, otherwise):
    """A conditional for OmegaConf interpolations."""
    if a == b:
        return then
    else:
        return otherwise


def resolve_eval(expr):
    """Resolve an arbitrary expression in OmegaConf interpolations."""
    # We trust our own configuration not to delete all our files, so just eval the
    # expression
    return eval(expr, {}, {"math": math})


OmegaConf.register_new_resolver("if_eq", if_eq)
OmegaConf.register_new_resolver("eval", resolve_eval)


log = get_logger()


def store_slurm_job_id(config: DictConfig):
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")

    with open_dict(config):
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id


class Null_SignalConnector(_SignalConnector):
    def register_signal_handlers(self):
        pass


def get_callbacks(config):
    if config.monitor is not None:
        monitor = {"monitor": config.monitor, "mode": "min"}
    else:
        monitor = {}
    callbacks = [
        WandbModelCheckpoint(
            dirpath=Path(config.checkpoint_root) / f"seed_{config.seed}",
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            filename="best",
            **monitor,
        ),
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor(logging_interval="step"),
        OpenFOAMPlots(data_dir=Path(config.data.root) / "data"),
        ConfigInCheckpoint(config),
    ]
    if monitor != {}:
        callbacks.append(WandbSummaries(**monitor))
    if config.get("early_stopping") is not None and monitor != {}:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    if config.get("train_limit") is not None:
        callbacks.append(TimeLimit(config.train_limit))
    return callbacks


def save_test_results(config: DictConfig, test_results: list[dict]):
    """Store this seed's test metrics and refresh the cross-seed summary files."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        key: float(value.item()) if torch.is_tensor(value) else float(value)
        for result in test_results
        for key, value in result.items()
    }
    record = {"seed": str(config.seed), **metrics}
    result_path = results_dir / f"seed_{config.seed}.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    records = [json.loads(path.read_text()) for path in sorted(results_dir.glob("seed_*.json"))]
    (results_dir / "summary.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    fieldnames = ["seed"] + sorted(
        {key for item in records for key in item if key != "seed"}
    )
    with (results_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    log.info(f"Test results saved to {result_path}")


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    set_seed(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    wandb.init(**config.wandb, resume=(config.wandb.mode == "online") and "allow")
    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)
    

    log.info("Instantiating data and system")
    datamodule, task = instantiate_data_and_task(config)
    if config.get("compile") is not None:
        log.info("Compiling model")
        task = torch.compile(task, mode=config.compile)

    logger = WandbLogger()
    log_hyperparameters(logger, config, task)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config)
    trainer: Trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
    # submitit handles the requeuing, so we disable pytorch-lightning's SLURM feature
    trainer.signal_connector = Null_SignalConnector(trainer)

    if config.get("restart_from") is not None:
        log.info(f"Restarting training from {config.restart_from}")
        ckpt_path = config.restart_from
    else:
        ckpt_path = None

    log.info("Starting training!")  
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
    
    if config.eval_testset:
        log.info("Starting testing!")
        test_results = trainer.test(ckpt_path="best", datamodule=datamodule)
        if trainer.global_rank == 0:
            save_test_results(config, test_results)

    # wandb.finish()
    if trainer.global_rank == 0:
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
