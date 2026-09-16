#!/usr/bin/env python3
"""Visualize generated 3-D flow samples against reference LES snapshots.

Examples
--------
Visualize every case in one evaluation file on the central x-z plane::

    python scripts/visualize-generated-samples.py \
        /data1/turbdiff/outputs/wogeo/samples/seed_2883413570083077179.h5

Select cases, a sample and a transverse y-z plane::

    python scripts/visualize-generated-samples.py SAMPLES.h5 \
        --cases cylinder torus2 --sample-index 4 --plane yz --slice-frac 0.65

The reference LES index follows OpenFOAMEvaluationSampler: after discarding early
times, ``n_samples`` indices are selected uniformly over the remaining series.
Because this diffusion model is not conditioned on time, a generated realization
is not pointwise paired with that LES snapshot.  Single-sample differences are
therefore qualitative; ensemble mean and standard-deviation plots are the more
meaningful statistical comparison.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


PLANE_AXIS = {"yz": 0, "xz": 1, "xy": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="SampleStore .h5 produced by eval_ckpt.py")
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data1/turbdiff/shapes/data"),
        help="Directory containing <case>/data.h5",
    )
    parser.add_argument("--cases", nargs="*", help="Cases to render (default: all in sample file)")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--plane", choices=tuple(PLANE_AXIS), default="xz")
    parser.add_argument(
        "--slice-frac", type=float, default=0.5,
        help="Slice position as a fraction in [0, 1] along the plane-normal axis",
    )
    parser.add_argument(
        "--discard-first-seconds", type=float, default=0.025,
        help="Must match data.discard_first_seconds used during evaluation",
    )
    parser.add_argument("--montage-count", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=Path("results/generated-visualizations"),
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def robust_limits(arrays: list[np.ndarray], *, symmetric: bool = False) -> tuple[float, float]:
    values = np.concatenate([x[np.isfinite(x)].ravel() for x in arrays])
    if not len(values):
        return (0.0, 1.0)
    if symmetric:
        bound = float(np.percentile(np.abs(values), 99.0))
        return (-bound, bound) if bound > 0 else (-1.0, 1.0)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def save(fig: plt.Figure, output: Path, stem: str, formats: list[str], dpi: int) -> None:
    for extension in formats:
        fig.savefig(output / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


class CaseData:
    def __init__(self, samples_file: Path, les_file: Path, case: str, discard: float):
        self.samples_file = samples_file
        self.les_file = les_file
        self.case = case
        self.discard = discard

        with h5py.File(les_file, "r") as handle:
            self.shape = tuple(int(x) for x in handle["grid/cell_counts"][:])
            self.cell_idx = np.asarray(handle["grid/cell_idx"][:], dtype=np.int64)
            self.cell_type = np.asarray(handle["grid/cell_type"][:])
            self.times = np.asarray(handle["data/times"][:])
        self.cell_coords = np.unravel_index(self.cell_idx, self.shape)
        with h5py.File(samples_file, "r") as handle:
            group = handle[f"{case}/data"]
            self.n_samples = int(group.attrs.get("n_samples", group["u"].shape[0]))

        valid = np.flatnonzero(self.times > discard)
        if not len(valid):
            raise ValueError(f"No LES times remain after discard={discard} for {case}")
        positions = np.round(np.linspace(0, len(valid) - 1, self.n_samples)).astype(int)
        self.les_indices = valid[positions]

    def read_generated(self, variable: str, indices=None) -> np.ndarray:
        with h5py.File(self.samples_file, "r") as handle:
            dataset = handle[f"{self.case}/data/{variable}"]
            if indices is None:
                return np.asarray(dataset[: self.n_samples])
            if np.ndim(indices) > 0:
                indices = np.asarray(indices).tolist()
            else:
                indices = int(indices)
            return np.asarray(dataset[indices])

    def read_les(self, variable: str, sample_indices=None) -> np.ndarray:
        if sample_indices is None:
            indices = self.les_indices.tolist()
        else:
            indices = self.les_indices[np.asarray(sample_indices)]
            if np.ndim(indices) > 0:
                indices = np.asarray(indices).tolist()
            else:
                indices = int(indices)
        # h5py fancy indexing requires increasing indices. The evaluation mapping
        # is increasing, and scalar indexing also works directly.
        with h5py.File(self.les_file, "r") as handle:
            return np.asarray(handle[f"data/{variable}"][indices])

    def plane(self, values: np.ndarray, plane: str, fraction: float) -> np.ndarray:
        """Map cell data directly onto one 2-D plane without building a 3-D grid."""
        values = np.asarray(values)
        if values.ndim >= 2 and values.shape[-1] == 1:
            values = values[..., 0]
        is_vector = values.ndim >= 2 and values.shape[-2] == len(self.cell_idx)
        if is_vector:
            batch_shape = values.shape[:-2]
            channels = (values.shape[-1],)
        else:
            if values.shape[-1] != len(self.cell_idx):
                raise ValueError(f"Unexpected cell-data shape: {values.shape}")
            batch_shape = values.shape[:-1]
            channels = ()

        axis = PLANE_AXIS[plane]
        plane_index = int(round((self.shape[axis] - 1) * fraction))
        selected = (self.cell_coords[axis] == plane_index) & (self.cell_type != 1)
        remaining_axes = [dim for dim in range(3) if dim != axis]
        plane_shape = tuple(self.shape[dim] for dim in remaining_axes)
        plane_idx = np.ravel_multi_index(
            tuple(self.cell_coords[dim][selected] for dim in remaining_axes), plane_shape
        )
        result = np.full((*batch_shape, *plane_shape, *channels), np.nan, dtype=np.float32)
        flat = result.reshape(*batch_shape, int(np.prod(plane_shape)), *channels)
        if channels:
            flat[..., plane_idx, :] = values[..., selected, :]
        else:
            flat[..., plane_idx] = values[..., selected]
        return flat.reshape(*batch_shape, *plane_shape, *channels)


def speed(u: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u, axis=-1)


def image(ax: plt.Axes, value: np.ndarray, norm, cmap: str, title: str) -> None:
    shown = ax.imshow(value.T, origin="lower", interpolation="none", norm=norm, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return shown


def comparison_figure(case: CaseData, index: int, plane: str, fraction: float) -> plt.Figure:
    gen_u = case.read_generated("u", index)
    les_u = case.read_les("u", index)
    gen_p = case.read_generated("p", index)
    les_p = case.read_les("p", index)

    fields = [
        ("Velocity magnitude", case.plane(speed(les_u), plane, fraction),
         case.plane(speed(gen_u), plane, fraction), "viridis", False),
        ("Pressure", case.plane(les_p, plane, fraction),
         case.plane(gen_p, plane, fraction), "coolwarm", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.7), constrained_layout=True)
    for row, (name, truth, generated, cmap, symmetric) in enumerate(fields):
        difference = np.abs(generated - truth)
        vmin, vmax = robust_limits([truth, generated], symmetric=symmetric)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        error_max = np.nanpercentile(difference, 99.0)
        error_norm = colors.Normalize(vmin=0, vmax=error_max if error_max > 0 else 1)
        im = image(axes[row, 0], truth, norm, cmap, f"LES {name}")
        image(axes[row, 1], generated, norm, cmap, f"Generated {name}")
        err = image(axes[row, 2], difference, error_norm, "magma", "Absolute difference")
        fig.colorbar(im, ax=axes[row, 1], shrink=0.82, pad=0.02)
        fig.colorbar(err, ax=axes[row, 2], shrink=0.82, pad=0.02)
    les_idx = int(case.les_indices[index])
    fig.suptitle(
        f"{case.case}: generated sample {index} vs reference LES index {les_idx} "
        f"(t={case.times[les_idx]:.4g}, {plane} plane at {fraction:.0%})",
        fontsize=13,
    )
    return fig


def ensemble_figure(case: CaseData, plane: str, fraction: float) -> plt.Figure:
    gen_u = case.plane(speed(case.read_generated("u")), plane, fraction)
    les_u = case.plane(speed(case.read_les("u")), plane, fraction)
    gen_p = case.plane(case.read_generated("p"), plane, fraction)
    les_p = case.plane(case.read_les("p"), plane, fraction)

    fields = [("Velocity magnitude", les_u, gen_u, "viridis", False),
              ("Pressure", les_p, gen_p, "coolwarm", True)]
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 6.2), constrained_layout=True)
    for row, (name, truth, generated, cmap, symmetric) in enumerate(fields):
        # All-NaN locations are outside the fluid mesh. Suppress NumPy's expected
        # empty-slice warnings while keeping these locations masked in the image.
        with np.errstate(invalid="ignore", divide="ignore"):
            truth_count = np.sum(np.isfinite(truth), axis=0)
            gen_count = np.sum(np.isfinite(generated), axis=0)
            truth_mean = np.nansum(truth, axis=0) / truth_count
            gen_mean = np.nansum(generated, axis=0) / gen_count
            truth_std = np.sqrt(
                np.nansum((truth - truth_mean) ** 2, axis=0)
                / np.maximum(truth_count - 1, 1)
            )
            gen_std = np.sqrt(
                np.nansum((generated - gen_mean) ** 2, axis=0)
                / np.maximum(gen_count - 1, 1)
            )
            truth_std[truth_count == 0] = np.nan
            gen_std[gen_count == 0] = np.nan
        mean_error = np.abs(gen_mean - truth_mean)

        mean_min, mean_max = robust_limits([truth_mean, gen_mean], symmetric=symmetric)
        mean_norm = colors.Normalize(mean_min, mean_max)
        std_min, std_max = robust_limits([truth_std, gen_std])
        std_norm = colors.Normalize(0, std_max)
        error_max = np.nanpercentile(mean_error, 99.0)
        error_norm = colors.Normalize(0, error_max if error_max > 0 else 1)
        im_mean = image(axes[row, 0], truth_mean, mean_norm, cmap, f"LES mean {name}")
        image(axes[row, 1], gen_mean, mean_norm, cmap, f"Generated mean {name}")
        im_error = image(axes[row, 2], mean_error, error_norm, "magma", "Mean-field error")
        im_std = image(axes[row, 3], truth_std, std_norm, "cividis", "LES temporal std.")
        image(axes[row, 4], gen_std, std_norm, "cividis", "Generated ensemble std.")
        fig.colorbar(im_mean, ax=axes[row, 1], shrink=0.75, pad=0.01)
        fig.colorbar(im_error, ax=axes[row, 2], shrink=0.75, pad=0.01)
        fig.colorbar(im_std, ax=axes[row, 4], shrink=0.75, pad=0.01)
    fig.suptitle(
        f"{case.case}: ensemble statistics from {case.n_samples} samples "
        f"({plane} plane at {fraction:.0%})",
        fontsize=13,
    )
    return fig


def montage_figure(case: CaseData, count: int, plane: str, fraction: float) -> plt.Figure:
    count = min(max(count, 1), case.n_samples)
    indices = np.linspace(0, case.n_samples - 1, count).round().astype(int)
    generated = case.plane(speed(case.read_generated("u", indices)), plane, fraction)
    slices = list(generated)
    vmin, vmax = robust_limits(slices)
    norm = colors.Normalize(vmin, vmax)
    ncols = min(3, count)
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.1 * nrows), squeeze=False,
                             constrained_layout=True)
    for ax, index, value in zip(axes.flat, indices, slices):
        shown = image(ax, value, norm, "viridis", f"Generated sample {index}")
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    fig.colorbar(shown, ax=list(axes.flat[:count]), shrink=0.75, label="Velocity magnitude")
    fig.suptitle(f"{case.case}: stochastic sample diversity ({plane} plane at {fraction:.0%})")
    return fig


def main() -> None:
    args = parse_args()
    if not args.samples.is_file():
        raise FileNotFoundError(args.samples)
    if not 0 <= args.slice_frac <= 1:
        raise ValueError("--slice-frac must be between 0 and 1")
    with h5py.File(args.samples, "r") as handle:
        available = sorted(handle.keys())
    selected = args.cases or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Cases not present in sample file: {', '.join(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)

    for name in selected:
        case = CaseData(
            args.samples, args.data_root / name / "data.h5", name,
            args.discard_first_seconds,
        )
        index = args.sample_index % case.n_samples
        prefix = f"{name}_{args.plane}_{args.slice_frac:.2f}"
        save(comparison_figure(case, index, args.plane, args.slice_frac), args.output,
             f"{prefix}_sample_{index}_comparison", args.formats, args.dpi)
        save(ensemble_figure(case, args.plane, args.slice_frac), args.output,
             f"{prefix}_ensemble", args.formats, args.dpi)
        save(montage_figure(case, args.montage_count, args.plane, args.slice_frac), args.output,
             f"{prefix}_diversity", args.formats, args.dpi)
        print(f"Rendered {name}: {case.n_samples} generated samples")
    print(f"Figures written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
