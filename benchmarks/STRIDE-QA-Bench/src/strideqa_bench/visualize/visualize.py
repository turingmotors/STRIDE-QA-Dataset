from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_prediction_distribution_plots(
    raw_results_list: list[dict[str, Any]], output_dir: str | Path
) -> None:
    """
    Create distribution plots for velocity, direction, and distance predictions with t0-t3 stacked vertically.

    Args:
        raw_results_list: List of raw evaluation results
        output_dir: Directory to save the plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Separate data by qa_type and time
    data_by_type_and_time = {
        "velocity": {0: [], 1: [], 2: [], 3: []},
        "direction": {0: [], 1: [], 2: [], 3: []},
        "distance": {0: [], 1: [], 2: [], 3: []},
    }

    for result in raw_results_list:
        qa_type = result["qa_type"]
        time = result["time"]
        if qa_type in data_by_type_and_time and time in data_by_type_and_time[qa_type]:
            data_by_type_and_time[qa_type][time].append(result)

    # Create stacked plots for each qa_type
    for qa_type, time_data in data_by_type_and_time.items():
        _create_stacked_time_plot(time_data, qa_type, output_dir)


def _create_stacked_time_plot(
    time_data: dict[int, list[dict[str, Any]]], qa_type: str, output_dir: Path
) -> None:
    """Create a stacked plot with t0-t3 arranged vertically for a specific qa_type."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    # Ensure axes is always 2D
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    time_steps = [0, 1, 2, 3]

    for i, time in enumerate(time_steps):
        data = time_data.get(time, [])

        # Left subplot: Value Distributions
        ax_dist = axes[i, 0]
        if data:
            _plot_distributions_subplot(data, qa_type, ax_dist)
            ax_dist.set_title(f"t={time}s - Value Distributions (n={len(data)})")
        else:
            ax_dist.text(
                0.5,
                0.5,
                f"t={time}s - No data available",
                ha="center",
                va="center",
                transform=ax_dist.transAxes,
            )
            ax_dist.set_title(f"t={time}s - Value Distributions")

        # Right subplot: GT vs Prediction
        ax_scatter = axes[i, 1]
        if data:
            _plot_scatter_subplot(data, qa_type, ax_scatter)
            ax_scatter.set_title(f"t={time}s - GT vs Prediction (n={len(data)})")
        else:
            ax_scatter.text(
                0.5,
                0.5,
                f"t={time}s - No data available",
                ha="center",
                va="center",
                transform=ax_scatter.transAxes,
            )
            ax_scatter.set_title(f"t={time}s - GT vs Prediction")

    plt.suptitle(
        f"{qa_type.title()} Analysis - All Time Steps",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    output_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{qa_type}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_distributions_subplot(data: list[dict[str, Any]], qa_type: str, ax: plt.Axes) -> None:
    """Plot prediction and GT value distributions in a subplot."""
    pred_values = []
    gt_values = []

    for result in data:
        if result["pred_value"] is not None:
            pred_values.append(result["pred_value"])
        if result["gt_value"] is not None:
            gt_values.append(result["gt_value"])

    if pred_values:
        ax.hist(
            pred_values,
            bins=20,
            alpha=0.7,
            color="skyblue",
            label="Predictions",
            edgecolor="black",
        )
    if gt_values:
        ax.hist(
            gt_values,
            bins=20,
            alpha=0.7,
            color="lightgreen",
            label="Ground Truth",
            edgecolor="black",
        )

    ax.set_title("Value Distributions")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set consistent aspect ratio for visual uniformity
    ax.set_aspect("auto", adjustable="box")


def _plot_scatter_subplot(data: list[dict[str, Any]], qa_type: str, ax: plt.Axes) -> None:
    """Plot GT vs Pred scatter in a subplot."""
    gt_values = []
    pred_values = []
    success_flags = []

    for result in data:
        if result["gt_value"] is not None and result["pred_value"] is not None:
            gt_values.append(result["gt_value"])
            pred_values.append(result["pred_value"])
            success_flags.append(result["success"])

    if not gt_values:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return

    # Separate successful and failed predictions
    gt_success = [gt for gt, success in zip(gt_values, success_flags) if success]
    pred_success = [pred for pred, success in zip(pred_values, success_flags) if success]
    gt_fail = [gt for gt, success in zip(gt_values, success_flags) if not success]
    pred_fail = [pred for pred, success in zip(pred_values, success_flags) if not success]

    if gt_success and pred_success:
        ax.scatter(
            gt_success,
            pred_success,
            c="green",
            alpha=0.6,
            label=f"Success ({len(gt_success)})",
            s=30,
        )
    if gt_fail and pred_fail:
        ax.scatter(
            gt_fail,
            pred_fail,
            c="red",
            alpha=0.6,
            label=f"Failure ({len(gt_fail)})",
            s=30,
        )

    # Perfect prediction line
    min_val = min(min(gt_values), min(pred_values))
    max_val = max(max(gt_values), max(pred_values))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Perfect Prediction",
    )

    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title("GT vs Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio for proper visualization of the perfect prediction line
    ax.set_aspect("equal", adjustable="box")

    # Ensure both axes have the same limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)


def create_lsr_analysis_plots(
    raw_results_list: list[dict[str, Any]], output_dir: str | Path
) -> None:
    """
    Create specialized visualization plots for LSR (Localization Success Rate) analysis.

    Args:
        raw_results_list: List of raw evaluation results
        output_dir: Directory to save the plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter LSR data
    lsr_data = [result for result in raw_results_list if result["qa_type"] == "lsr"]

    if not lsr_data:
        print("No LSR data found for visualization.")
        return

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create different types of LSR analysis plots
    _create_polar_plot_analysis(lsr_data, output_dir)
    _create_error_analysis_plots(lsr_data, output_dir)
    _create_lsr_comparison_plots(raw_results_list, output_dir)


def _create_polar_plot_analysis(lsr_data: list[dict[str, Any]], output_dir: Path) -> None:
    """Create polar coordinate plots for LSR visualization."""
    # Create horizontal layout for publication quality (1x4 but more compact)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), subplot_kw=dict(projection="polar"))

    time_steps = [0, 1, 2, 3]

    # Set fixed distance scale for all subplots (0-50 meters)
    unified_max_distance = 50

    for i, time in enumerate(time_steps):
        ax = axes[i]
        time_data = [result for result in lsr_data if result["time"] == time]

        if not time_data:
            ax.text(
                0.5,
                0.5,
                f"t={time}s - No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            continue

        # Set polar plot orientation: 0 degrees at the top (North)
        ax.set_theta_zero_location("N")  # 0 degrees at the top (front direction)
        ax.set_theta_direction(1)  # Counterclockwise for positive angles (correct direction)

        # Set angle range to -180 to +180 degrees (matching data definition)
        ax.set_thetalim(-np.pi, np.pi)  # Set range to -π to +π radians (-180° to +180°)

        # Set custom angle labels: exclude -180 and -90 to avoid overlap
        angle_ticks = np.array([-135, -45, 0, 45, 90, 135, 180])
        angle_labels = [f"{int(angle)}°" for angle in angle_ticks]
        ax.set_thetagrids(angle_ticks, angle_labels)

        # Adjust font sizes for publication quality (increased)
        ax.tick_params(axis="both", which="major", labelsize=12)  # Angle labels

        # Position radial (distance) labels on the right side (-90 degrees) to avoid overlap with data
        ax.set_rlabel_position(-90)

        # Add front camera FOV visualization (hFOV = 60°, ±30° from center)
        fov_half_angle = 30  # degrees
        fov_theta = np.linspace(-fov_half_angle * np.pi / 180, fov_half_angle * np.pi / 180, 100)
        fov_radius = np.full_like(fov_theta, unified_max_distance)

        # Fill the FOV area with a light color
        ax.fill_between(
            fov_theta,
            0,
            fov_radius,
            alpha=0.15,
            color="blue",
            edgecolor="none",
            label=f"Camera FOV (±{fov_half_angle}°)",
        )

        # Extract distance and direction data with error calculation
        distances_success = []
        directions_success = []
        sizes_success = []
        distances_fail = []
        directions_fail = []
        sizes_fail = []

        # Calculate errors for all valid data points
        all_errors = []

        # Debug counters for t=1
        none_skipped_count = 0
        valid_count = 0

        for result in time_data:
            distance_gt = result["gt_value"]["distance"]
            distance_pred = result["pred_value"]["distance"]
            direction_gt = result["gt_value"]["direction"]
            direction_pred = result["pred_value"]["direction"]

            # Skip entries with None values
            if (
                distance_gt is None
                or distance_pred is None
                or direction_gt is None
                or direction_pred is None
            ):
                if time == 1:
                    none_skipped_count += 1
                continue

            if time == 1:
                valid_count += 1

            # Calculate relative distance error
            if distance_gt > 0:
                dist_error = abs(distance_pred - distance_gt) / distance_gt
            else:
                dist_error = abs(distance_pred - distance_gt)

            # Calculate angular error (considering wraparound)
            dir_error = abs((direction_pred - direction_gt + 180) % 360 - 180)

            # Combine errors (normalize direction error to 0-1 scale)
            combined_error = dist_error + (dir_error / 180.0)  # Normalize dir_error to 0-1
            all_errors.append(combined_error)

        # Normalize errors to size range (publication optimized)
        if all_errors:
            min_error = min(all_errors)
            max_error = max(all_errors)
            error_range = max_error - min_error if max_error > min_error else 1

        for result in time_data:
            distance_gt = result["gt_value"]["distance"]
            distance_pred = result["pred_value"]["distance"]
            direction_gt = result["gt_value"]["direction"]
            direction_pred = result["pred_value"]["direction"]

            # Skip entries with None values
            if (
                distance_gt is None
                or distance_pred is None
                or direction_gt is None
                or direction_pred is None
            ):
                continue

            # Calculate errors
            if distance_gt > 0:
                dist_error = abs(distance_pred - distance_gt) / distance_gt
            else:
                dist_error = abs(distance_pred - distance_gt)

            dir_error = abs((direction_pred - direction_gt + 180) % 360 - 180)
            combined_error = dist_error + (dir_error / 180.0)

            # Map error to size (optimized for publication)
            if all_errors and error_range > 0:
                normalized_error = (combined_error - min_error) / error_range
                point_size = 12 + normalized_error * 60  # Size range: 12-72 pixels
            else:
                point_size = 30  # Default size

            # Convert direction to radians
            direction_rad = np.radians(direction_pred)

            if result["success"]:
                distances_success.append(distance_pred)
                directions_success.append(direction_rad)
                sizes_success.append(point_size)
            else:
                distances_fail.append(distance_pred)
                directions_fail.append(direction_rad)
                sizes_fail.append(point_size)

        # Plot successful predictions with error-based sizing
        if distances_success:
            ax.scatter(
                directions_success,
                distances_success,
                c="green",
                alpha=0.8,
                s=sizes_success,
                label=f"Success ({len(distances_success)})",
            )

        # Plot failed predictions with error-based sizing
        if distances_fail:
            ax.scatter(
                directions_fail,
                distances_fail,
                c="red",
                alpha=0.8,
                s=sizes_fail,
                label=f"Failure ({len(distances_fail)})",
            )

        # Set title with larger font size for publication
        ax.set_title(f"t = {time}s", fontsize=16, fontweight="bold", pad=15)

        # Use fixed distance scale (0-50 meters) for all subplots
        ax.set_ylim(0, unified_max_distance)

        # Add legend for each subplot with individual Success/Failure counts
        # Position at lower left to avoid data overlap
        legend_elements = []
        if distances_success:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    label=f"Success ({len(distances_success)})",
                )
            )
        if distances_fail:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    label=f"Failure ({len(distances_fail)})",
                )
            )

        # Add Camera FOV legend only for t=0 since it's common for all subplots
        if i == 0:  # Only for t=0
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor="blue", alpha=0.15, label="Camera FOV (±30°)")
            )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="lower left",
                bbox_to_anchor=(0.02, 0.02),
                fontsize=11,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        ax.grid(True, alpha=0.3)

        # Adjust radial tick labels font size (increased)
        ax.tick_params(axis="y", labelsize=12)

    # Remove main title for publication quality (not needed in papers)
    # Optimize layout for publication quality with more compact spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.92, wspace=0.15)
    output_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "lsr_polar.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def _create_error_analysis_plots(lsr_data: list[dict[str, Any]], output_dir: Path) -> None:
    """Create error analysis plots for LSR data."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    time_steps = [0, 1, 2, 3]

    for i, time in enumerate(time_steps):
        ax = axes[i // 2, i % 2]
        time_data = [result for result in lsr_data if result["time"] == time]

        if not time_data:
            ax.text(
                0.5, 0.5, f"t={time}s - No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Calculate errors
        distance_errors = []
        direction_errors = []
        success_flags = []

        for result in time_data:
            distance_gt = result["gt_value"]["distance"]
            distance_pred = result["pred_value"]["distance"]
            direction_gt = result["gt_value"]["direction"]
            direction_pred = result["pred_value"]["direction"]

            # Skip entries with None values
            if (
                distance_gt is None
                or distance_pred is None
                or direction_gt is None
                or direction_pred is None
            ):
                continue

            # Calculate relative distance error
            if distance_gt > 0:
                dist_error = abs(distance_pred - distance_gt) / distance_gt
            else:
                dist_error = abs(distance_pred - distance_gt)

            # Calculate angular error (considering wraparound)
            dir_error = abs((direction_pred - direction_gt + 180) % 360 - 180)

            distance_errors.append(dist_error)
            direction_errors.append(dir_error)
            success_flags.append(result["success"])

        # Skip if no valid data after filtering
        if not distance_errors:
            ax.text(
                0.5,
                0.5,
                f"t={time}s - No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Separate successful and failed predictions
        dist_err_success = [de for de, success in zip(distance_errors, success_flags) if success]
        dir_err_success = [de for de, success in zip(direction_errors, success_flags) if success]
        dist_err_fail = [de for de, success in zip(distance_errors, success_flags) if not success]
        dir_err_fail = [de for de, success in zip(direction_errors, success_flags) if not success]

        # Plot error scatter
        if dist_err_success and dir_err_success:
            ax.scatter(
                dist_err_success,
                dir_err_success,
                c="green",
                alpha=0.6,
                s=30,
                label=f"Success ({len(dist_err_success)})",
            )
        if dist_err_fail and dir_err_fail:
            ax.scatter(
                dist_err_fail,
                dir_err_fail,
                c="red",
                alpha=0.6,
                s=30,
                label=f"Failure ({len(dist_err_fail)})",
            )

        # Add tolerance lines
        ax.axvline(0.25, color="blue", linestyle="--", alpha=0.7, label="Distance Tolerance (25%)")
        ax.axhline(10, color="orange", linestyle="--", alpha=0.7, label="Direction Tolerance (10°)")

        ax.set_xlabel("Distance Relative Error")
        ax.set_ylabel("Direction Absolute Error (degrees)")
        ax.set_title(f"Error Analysis - t={time}s")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "LSR Error Analysis - Distance vs Direction Errors", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    output_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "lsr_error.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def _create_lsr_comparison_plots(raw_results_list: list[dict[str, Any]], output_dir: Path) -> None:
    """Create comparison plots between individual metrics and LSR."""
    # Calculate success rates by time for different metrics
    metrics_data = {"target_distance": [], "target_direction": [], "lsr": []}

    time_steps = [0, 1, 2, 3]

    for time in time_steps:
        # Filter data for each metric at current time
        distance_data = [
            r
            for r in raw_results_list
            if r["time"] == time and r["qa_type"] == "distance" and r["ego_or_target"] == "target"
        ]
        direction_data = [
            r
            for r in raw_results_list
            if r["time"] == time and r["qa_type"] == "direction" and r["ego_or_target"] == "target"
        ]
        lsr_data = [r for r in raw_results_list if r["time"] == time and r["qa_type"] == "lsr"]

        # Calculate success rates
        dist_success_rate = (
            sum(r["success"] for r in distance_data) / len(distance_data) if distance_data else 0
        )
        dir_success_rate = (
            sum(r["success"] for r in direction_data) / len(direction_data) if direction_data else 0
        )
        lsr_success_rate = sum(r["success"] for r in lsr_data) / len(lsr_data) if lsr_data else 0

        metrics_data["target_distance"].append(dist_success_rate)
        metrics_data["target_direction"].append(dir_success_rate)
        metrics_data["lsr"].append(lsr_success_rate)

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Success rate comparison
    x = np.arange(len(time_steps))
    width = 0.25

    ax1.bar(x - width, metrics_data["target_distance"], width, label="Distance Only", alpha=0.8)
    ax1.bar(x, metrics_data["target_direction"], width, label="Direction Only", alpha=0.8)
    ax1.bar(x + width, metrics_data["lsr"], width, label="LSR (Combined)", alpha=0.8)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate Comparison: Individual vs Combined (LSR)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"t={t}" for t in time_steps])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for i, time in enumerate(time_steps):
        ax1.text(
            i - width,
            metrics_data["target_distance"][i] + 0.01,
            f"{metrics_data['target_distance'][i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax1.text(
            i,
            metrics_data["target_direction"][i] + 0.01,
            f"{metrics_data['target_direction'][i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax1.text(
            i + width,
            metrics_data["lsr"][i] + 0.01,
            f"{metrics_data['lsr'][i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Performance degradation over time
    ax2.plot(
        time_steps,
        metrics_data["target_distance"],
        "o-",
        label="Distance Only",
        linewidth=2,
        markersize=8,
    )
    ax2.plot(
        time_steps,
        metrics_data["target_direction"],
        "s-",
        label="Direction Only",
        linewidth=2,
        markersize=8,
    )
    ax2.plot(
        time_steps,
        metrics_data["lsr"],
        "^-",
        label="LSR (Combined)",
        linewidth=2,
        markersize=8,
    )

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Performance Degradation Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(time_steps)
    ax2.set_xticklabels([f"t={t}" for t in time_steps])

    plt.tight_layout()
    output_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "lsr_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Model-specific visualization settings
# ----------------------------------------------------------------------
MODEL_SPECS: dict[str, dict[str, str]] = {
    "GPT-4o": {
        "label": "GPT-4o",
        "color": "#7f7f7f",  # gray
        "marker": "o",
    },
    "GPT-4o-mini": {
        "label": "GPT-4o-mini",
        "color": "#e377c2",  # pink
        "marker": "s",
    },
    "Qwen2.5-VL-7B-Instruct": {
        "label": "Qwen2.5-VL-7B",
        "color": "#2ca02c",  # green
        "marker": "D",
    },
    "InternVL2_5-8B": {
        "label": "InternVL2.5-8B",
        "color": "#9467bd",  # purple
        "marker": "^",
    },
    "Cosmos-Reason1-7B": {
        "label": "Cosmos-Reason1-7B",
        "color": "#ff7f0e",  # orange
        "marker": "v",
    },
    "Senna-VLM": {
        "label": "Senna-VLM",
        "color": "#8c564b",  # brown
        "marker": "X",
    },
    "STRIDE-Qwen2.5-VL-7B-Instruct": {
        "label": "STRIDE-Qwen2.5-VL-7B",
        "color": "#5fc5dc",  # blue
        "marker": "*",  # star
        "markersize": 10,  # larger
    },
    "STRIDE-Cosmos-Reason1-7B": {
        "label": "STRIDE-Cosmos-Reason1-7B",
        "color": "#388bb3",  # cyan
        "marker": "*",
        "markerface": "white",  # white
        "markersize": 10,  # larger
        "markeredgewidth": 1.0,
    },
}


def create_lsr_relation_plots(
    model_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    rel_thresh_minor: int = 15,
    json_name: str = "raw_evaluation.json",
    pdf_name: str = "lsr_relation_grid.pdf",
) -> Path:
    """Generate an LSR-time grid grouped by relations.

    Args:
        model_dirs: Mapping ``{model_id: result_dir}``.
        output_dir: Directory where the PDF will be saved.
        rel_thresh_minor: Threshold below which relations are grouped as minor.
        json_name: Evaluation file name inside each result directory.
        pdf_name: Output PDF file name.

    Returns:
        Path to the generated PDF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load ----------
    df_list = []
    for model_id, dir_path in model_dirs.items():
        json_path = Path(dir_path) / json_name
        with open(json_path, encoding="utf-8") as f:
            entries = json.load(f)
        df_m = pd.DataFrame(entries)
        df_m = df_m[df_m["qa_type"] == "lsr"].copy()
        df_m["model"] = model_id
        df_list.append(df_m)

    df = pd.concat(df_list, ignore_index=True)

    # ---------- 2. Merge Minor ----------
    major_relations = {
        "Oncoming_Pass",
        "Maintain_State",
        "Overtake",
        "Path_Divergence_at_Intersection",
        "Other_Pulling_Away_From_Ego",
    }
    rel_counts = df.groupby("relation", dropna=False)["success"].size()

    def to_minor(rel: str) -> str:
        if pd.isna(rel) or rel not in major_relations or rel_counts.get(rel, 0) <= rel_thresh_minor:
            return "Minor_Relations"
        return rel

    df["relation_merged"] = df["relation"].map(to_minor)

    # ---------- 3. Aggregate ----------
    agg = (
        df.groupby(["model", "relation_merged", "time"], sort=False)["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )

    time_steps = sorted(df["time"].unique())
    all_relations = df["relation_merged"].unique()
    complete_index = pd.MultiIndex.from_product(
        [model_dirs.keys(), all_relations, time_steps],
        names=["model", "relation_merged", "time"],
    )
    agg = (
        agg.set_index(["model", "relation_merged", "time"])
        .reindex(complete_index, fill_value=0.0)
        .reset_index()
    )

    # ---------- 4. Plot ----------
    first_model = next(iter(model_dirs))
    base_df = df[(df["model"] == first_model) & (df["time"] == time_steps[0])]
    rel_sample_count = base_df.groupby("relation_merged").size().to_dict()

    panel_order = [
        "Maintain_State",
        "Path_Divergence_at_Intersection",
        "Overtake",
        "Other_Pulling_Away_From_Ego",
        "Oncoming_Pass",
        "Minor_Relations",
    ]
    valid_relations = [r for r in panel_order if r in all_relations]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()

    # Background color grouping
    # left_bg_color = "#f5f5f5"
    left_bg_color = "#f0f0f0"
    right_bg_color = "#eaf2fa"
    left_group_indices = [0, 3]
    right_group_indices = [1, 2, 4, 5]
    for i in left_group_indices:
        if i < len(axes):
            axes[i].set_facecolor(left_bg_color)
    for i in right_group_indices:
        if i < len(axes):
            axes[i].set_facecolor(right_bg_color)
    axes[-1].set_facecolor("white")

    # Plot
    for idx, rel in enumerate(valid_relations):
        ax = axes[idx]
        for model_id in model_dirs:
            spec = MODEL_SPECS.get(model_id, {})
            y = (
                agg.query("model == @model_id and relation_merged == @rel")
                .sort_values("time")["success_rate"]
                .to_numpy()
            )
            ax.plot(
                time_steps,
                y,
                marker=spec.get("marker", "o"),
                lw=1.3,
                color=spec.get("color", "#333333"),
                label=spec.get("label", model_id),
                markerfacecolor=spec.get("markerface", spec.get("color", "#333333")),
                markersize=spec.get("markersize", 4),
                markeredgewidth=spec.get("markeredgewidth", 1),
            )

        n_samples = rel_sample_count.get(rel, 0)
        rel_to_title = {
            "Oncoming_Pass": "Oncoming Pass",
            "Maintain_State": "Maintain State",
            "Overtake": "Overtake",
            "Path_Divergence_at_Intersection": "Path Divergence",
            "Other_Pulling_Away_From_Ego": "Pulling Away From Ego",
            "Minor_Relations": "Minor Relations",
        }
        ax.set_title(f"{rel_to_title[rel]} (n={n_samples})", fontsize=11)
        ax.set_xticks(time_steps)
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.set_ylim(-0.05, 1.0)
        # ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel("LSR", fontsize=11)
        if idx // 3 == 1:
            ax.set_xlabel("Time (s)", fontsize=11)

    # Remove empty panels
    for j in range(len(valid_relations), len(axes)):
        fig.delaxes(axes[j])

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=MODEL_SPECS[m]["color"],
            marker=MODEL_SPECS[m]["marker"],
            markersize=MODEL_SPECS[m].get("markersize", 6),
            markeredgewidth=MODEL_SPECS[m].get("markeredgewidth", 1.2),
            markeredgecolor=MODEL_SPECS[m]["color"],
            markerfacecolor=MODEL_SPECS[m].get("markerface", MODEL_SPECS[m]["color"]),
            label=MODEL_SPECS[m]["label"],
        )
        for m in model_dirs
    ]

    fig.legend(
        handles=handles,
        ncol=min(4, len(handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    pdf_path = Path(output_dir) / pdf_name
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path
