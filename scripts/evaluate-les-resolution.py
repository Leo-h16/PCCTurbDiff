#!/usr/bin/env python

"""Evaluate LES resolution indicators stored in the generated HDF5 files.

This computes the resolved turbulent kinetic energy and compares it with the
SGS kinetic energy produced by OpenFOAM's dynamicKEqn model.  It is a
resolution diagnostic, not a replacement for a grid-convergence study or an
external validation case.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def accumulate(dataset, start, stop, time_chunk):
    """Accumulate first and second moments without loading all times."""
    n_cells = dataset.shape[1]
    trailing_shape = dataset.shape[2:]
    total = np.zeros((n_cells, *trailing_shape), dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = 0

    for begin in range(start, stop, time_chunk):
        end = min(begin + time_chunk, stop)
        values = np.asarray(dataset[begin:end], dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        count += end - begin

    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    return mean, variance


def mean_field(dataset, start, stop, time_chunk):
    n_cells = dataset.shape[1]
    total = np.zeros(n_cells, dtype=np.float64)
    count = 0
    for begin in range(start, stop, time_chunk):
        end = min(begin + time_chunk, stop)
        values = np.asarray(dataset[begin:end], dtype=np.float64)
        total += values.sum(axis=0)
        count += end - begin
    return total / count


def find_cell_volumes(file, n_cells):
    """Use cell volumes if a future/reprocessed HDF5 file contains them."""
    for key in ("domain/cell_volumes", "domain/volumes", "data/v"):
        if key not in file:
            continue
        values = np.asarray(file[key])
        if values.ndim == 2:
            values = values[0]
        values = values.reshape(-1).astype(np.float64)
        if len(values) == n_cells and np.all(values > 0):
            return values, key
    return np.ones(n_cells, dtype=np.float64), None


def weighted_mean(values, weights, mask):
    selected_weights = weights[mask]
    if selected_weights.size == 0 or selected_weights.sum() == 0:
        return float("nan")
    return float(np.sum(values[mask] * selected_weights) / selected_weights.sum())


def weighted_fraction(mask, weights, region):
    denominator = weights[region].sum()
    if denominator == 0:
        return float("nan")
    return float(weights[mask & region].sum() / denominator)


def quantiles(values, mask):
    selected = values[mask]
    if selected.size == 0:
        return {"q05": None, "q50": None, "q95": None}
    q05, q50, q95 = np.quantile(selected, (0.05, 0.5, 0.95))
    return {"q05": float(q05), "q50": float(q50), "q95": float(q95)}


def calculate_fields(file, start, stop, time_chunk):
    mean_u, variance_u = accumulate(file["data/u"], start, stop, time_chunk)
    k_resolved = 0.5 * variance_u.sum(axis=-1)
    k_sgs = mean_field(file["data/k"], start, stop, time_chunk)
    nut = mean_field(file["data/nut"], start, stop, time_chunk)
    return mean_u, k_resolved, k_sgs, nut


def evaluate_case(path, args):
    with h5py.File(path, "r") as file:
        required = ("data/times", "data/u", "data/k", "data/nut")
        missing = [key for key in required if key not in file]
        if missing:
            raise RuntimeError(f"{path}: missing datasets {missing}")

        times = np.asarray(file["data/times"])
        first_time = args.discard_first_seconds
        if args.start_time is not None:
            first_time = args.start_time
        start = int(np.searchsorted(times, first_time, side="left"))
        stop = len(times)
        if args.end_time is not None:
            stop = int(np.searchsorted(times, args.end_time, side="right"))
        if stop - start < max(args.blocks, 2):
            raise RuntimeError(f"{path}: only {stop - start} selected time samples")

        mean_u, k_resolved, k_sgs, nut = calculate_fields(
            file, start, stop, args.time_chunk
        )
        n_cells = len(k_resolved)
        weights, volume_key = find_cell_volumes(file, n_cells)
        nu = float(file["physical"].attrs["nu"])

        total_tke = k_resolved + k_sgs
        valid = (
            np.isfinite(mean_u).all(axis=-1)
            & np.isfinite(total_tke)
            & np.isfinite(nut)
            & (total_tke >= 0)
            & (nut >= 0)
        )
        positive = valid & (total_tke > np.finfo(np.float64).eps)
        if not np.any(positive):
            raise RuntimeError(f"{path}: no cells with positive turbulent energy")

        active_threshold = args.active_tke_fraction * np.max(total_tke[positive])
        active = valid & (total_tke >= active_threshold)
        resolved_fraction = np.divide(
            k_resolved,
            total_tke,
            out=np.full_like(total_tke, np.nan),
            where=positive,
        )
        resolved_above_threshold = np.zeros(n_cells, dtype=bool)
        np.greater_equal(
            resolved_fraction,
            args.resolved_threshold,
            out=resolved_above_threshold,
            where=positive,
        )
        nut_over_nu = nut / nu

        total_energy = np.sum(total_tke[active] * weights[active])
        resolved_energy_fraction = float(
            np.sum(k_resolved[active] * weights[active]) / total_energy
        )

        block_fractions = []
        for indices in np.array_split(np.arange(start, stop), args.blocks):
            block_start = int(indices[0])
            block_stop = int(indices[-1]) + 1
            _, block_k_resolved, block_k_sgs, _ = calculate_fields(
                file, block_start, block_stop, args.time_chunk
            )
            block_total = block_k_resolved + block_k_sgs
            denominator = np.sum(block_total[active] * weights[active])
            block_fractions.append(
                float(np.sum(block_k_resolved[active] * weights[active]) / denominator)
            )

        block_mean = float(np.mean(block_fractions))
        block_relative_range = float(
            (max(block_fractions) - min(block_fractions)) / block_mean
        )
        resolved_quantiles = quantiles(resolved_fraction, active & positive)
        nut_quantiles = quantiles(nut_over_nu, active)

        result = {
            "case": path.parent.name,
            "file": str(path.resolve()),
            "n_cells": n_cells,
            "n_invalid_cells": int(n_cells - np.count_nonzero(valid)),
            "n_time_samples": stop - start,
            "start_time": float(times[start]),
            "end_time": float(times[stop - 1]),
            "nu": nu,
            "weighting": "cell-volume" if volume_key else "equal-cell (approximate)",
            "cell_volume_dataset": volume_key,
            "active_tke_threshold": float(active_threshold),
            "active_cell_fraction": float(np.mean(active)),
            "resolved_energy_fraction_active": resolved_energy_fraction,
            "resolved_fraction_active_q05": resolved_quantiles["q05"],
            "resolved_fraction_active_q50": resolved_quantiles["q50"],
            "resolved_fraction_active_q95": resolved_quantiles["q95"],
            "active_weight_fraction_resolved_gt_0_8": weighted_fraction(
                resolved_above_threshold, weights, active
            ),
            "nut_over_nu_active_mean": weighted_mean(nut_over_nu, weights, active),
            "nut_over_nu_active_q05": nut_quantiles["q05"],
            "nut_over_nu_active_q50": nut_quantiles["q50"],
            "nut_over_nu_active_q95": nut_quantiles["q95"],
            "active_weight_fraction_nut_over_nu_gt_1": weighted_fraction(
                nut_over_nu > 1, weights, active
            ),
            "active_weight_fraction_nut_over_nu_gt_5": weighted_fraction(
                nut_over_nu > 5, weights, active
            ),
            "active_weight_fraction_nut_over_nu_gt_10": weighted_fraction(
                nut_over_nu > 10, weights, active
            ),
            "block_resolved_energy_fractions": block_fractions,
            "block_resolved_fraction_relative_range": block_relative_range,
            "passes_selected_resolution_heuristics": bool(
                resolved_energy_fraction >= args.resolved_threshold
                and weighted_fraction(
                    resolved_above_threshold, weights, active
                )
                >= args.coverage_threshold
                and block_relative_range <= args.max_block_relative_range
            ),
        }

        if args.save_fields:
            counts = tuple(np.asarray(file["grid/cell_counts"], dtype=int))
            cell_idx = np.asarray(file["grid/cell_idx"], dtype=np.int64)
            if len(cell_idx) == n_cells and cell_idx.max() < np.prod(counts):
                fields = {}
                for name, values in {
                    "k_resolved": k_resolved,
                    "k_sgs": k_sgs,
                    "resolved_fraction": resolved_fraction,
                    "nut_over_nu": nut_over_nu,
                    "active_mask": active,
                }.items():
                    grid = np.full(np.prod(counts), np.nan, dtype=np.float32)
                    grid[cell_idx] = values.astype(np.float32)
                    fields[name] = grid.reshape(counts)
                np.savez_compressed(
                    args.output_dir / f"{path.parent.name}-fields.npz",
                    **fields,
                    cell_counts=np.asarray(counts),
                )

    return result


def discover_files(dataset):
    if dataset.is_file():
        return [dataset]
    return sorted(dataset.rglob("data.h5"))


def write_summary(results, output_dir):
    scalar_keys = [
        key
        for key, value in results[0].items()
        if not isinstance(value, (list, dict))
    ]
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=scalar_keys)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in scalar_keys})
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root or one data.h5")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--discard-first-seconds", type=float, default=0.025)
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--time-chunk", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--active-tke-fraction", type=float, default=0.01)
    parser.add_argument("--resolved-threshold", type=float, default=0.8)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--max-block-relative-range", type=float, default=0.05)
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--cases", nargs="*", help="Only evaluate these case names")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()

    if args.time_chunk < 1 or args.blocks < 1:
        parser.error("--time-chunk and --blocks must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(args.dataset)
    if args.cases:
        names = set(args.cases)
        files = [path for path in files if path.parent.name in names]
    if args.max_cases is not None:
        files = files[: args.max_cases]
    if not files:
        parser.error("no data.h5 files found")

    results = []
    for index, path in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {path}", flush=True)
        try:
            result = evaluate_case(path, args)
        except Exception as error:
            result = {
                "case": path.parent.name,
                "file": str(path.resolve()),
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"  ERROR: {result['error']}", flush=True)
        else:
            print(
                "  resolved="
                f"{result['resolved_energy_fraction_active']:.4f}, "
                "coverage="
                f"{result['active_weight_fraction_resolved_gt_0_8']:.4f}, "
                "nut/nu(q50)="
                f"{result['nut_over_nu_active_q50']:.3f}",
                flush=True,
            )
            (args.output_dir / f"{path.parent.name}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        results.append(result)

    successful = [result for result in results if "error" not in result]
    if successful:
        write_summary(successful, args.output_dir)
        print(f"Summary: {args.output_dir / 'summary.csv'}")
    if len(successful) != len(results):
        (args.output_dir / "errors.json").write_text(
            json.dumps(
                [result for result in results if "error" in result],
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
