#!/usr/bin/env python3
"""
Compare polar plots across multiple models.

This script creates polar coordinate plots for LSR (Localization Success Rate) analysis
comparing multiple models side by side. Each row represents a model, and each column
represents a time step (t=0, t=1, t=2, t=3).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from strideqa_bench import STRIDEQA_BENCH_ROOT


def load_lsr_data_from_file(file_path: Path) -> list[dict[str, Any]]:
    """Load LSR data from raw_evaluation.json file."""
    with open(file_path, encoding="utf-8") as f:
        raw_results = json.load(f)

    # Filter LSR data
    lsr_data = [result for result in raw_results if result["qa_type"] == "lsr"]
    return lsr_data


def create_polar_plots_for_model(
    lsr_data: list[dict[str, Any]],
    axes: list[plt.Axes],
    model_name: str,
    row_idx: int,
    unified_max_distance: int = 50,
) -> None:
    """Create polar plots for a single model across 4 time steps."""
    time_steps = [0, 1, 2, 3]

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
                fontsize=13,
            )
            continue

        # Set polar plot orientation: 0 degrees at the top (North)
        ax.set_theta_zero_location("N")  # 0 degrees at the top (front direction)
        ax.set_theta_direction(1)  # Counterclockwise for positive angles (correct direction)

        # Set angle range to -180 to +180 degrees (matching data definition)
        ax.set_thetalim(-np.pi, np.pi)  # Set range to -π to +π radians (-180° to +180°)

        # Set custom angle labels: include -90 for grid but hide its label to avoid overlap
        angle_ticks = np.array([-135, -90, -45, 0, 45, 90, 135, 180])
        angle_labels = [f"{int(angle)}°" if angle != -90 else "" for angle in angle_ticks]
        ax.set_thetagrids(angle_ticks, angle_labels)

        # Adjust font sizes for publication quality (increased)
        ax.tick_params(axis="both", which="major", labelsize=13)  # Angle labels

        ax.set_rlabel_position(-112.5)

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

            point_size = 30  # Fixed size for all points

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
                alpha=0.7,
                s=sizes_success,
                label=f"Success ({len(distances_success)})",
            )

        # Plot failed predictions with error-based sizing
        if distances_fail:
            ax.scatter(
                directions_fail,
                distances_fail,
                c="red",
                alpha=0.7,
                s=sizes_fail,
                label=f"Failure ({len(distances_fail)})",
            )

        # Set title with larger font size for publication
        if row_idx == 0:  # Only add time labels on the top row
            ax.set_title(f"t = {time}s", fontsize=20, fontweight="bold", pad=15)

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

        # Camera FOV legend will be handled at figure level

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="lower left",
                bbox_to_anchor=(0.02, 0.02),
                fontsize=13,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        # Enhanced grid settings to ensure all radial and angular grid lines are visible
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_rgrids(range(0, unified_max_distance + 1, 10), alpha=0.7)  # Explicit radial grid

        # Adjust radial tick labels font size (increased)
        ax.tick_params(axis="y", labelsize=13)


def create_comparison_polar_plots(
    model_files: dict[str, Path],
    output_dir: Path,
    output_filename: str = "comparison_polar_plots.pdf",
) -> None:
    """
    Create comparison polar plots for multiple models.

    Args:
        model_files: Dictionary mapping model names to their raw_evaluation.json file paths
        output_dir: Output directory for the plots
        output_filename: Name of the output PDF file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for all models
    model_data = {}
    for model_name, file_path in model_files.items():
        print(f"Loading data for {model_name} from {file_path}")
        model_data[model_name] = load_lsr_data_from_file(file_path)

    num_models = len(model_data)
    num_time_steps = 4

    # Create figure with subplots
    fig_height = 4.3 * num_models
    fig, axes = plt.subplots(
        num_models, num_time_steps, figsize=(15, fig_height), subplot_kw=dict(projection="polar")
    )

    # Ensure axes is always 2D
    if num_models == 1:
        axes = axes.reshape(1, -1)
    elif num_time_steps == 1:
        axes = axes.reshape(-1, 1)

    # Create plots for each model
    for row_idx, (model_name, lsr_data) in enumerate(model_data.items()):
        print(f"Creating plots for {model_name}...")

        # Add model name as row label
        if num_models > 1:
            fig.text(
                0.04,
                1 - (row_idx + 0.5) / num_models - 0.02,
                model_name,
                rotation=90,
                va="center",
                ha="center",
                fontsize=18,
                fontweight="bold",
            )

        create_polar_plots_for_model(
            lsr_data=lsr_data,
            axes=axes[row_idx] if num_models > 1 else axes,
            model_name=model_name,
            row_idx=row_idx,
        )

        # Add figure-level legend for Camera FOV (common to all subplots)
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="blue", alpha=0.15, label="Camera FOV (±30°)")
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        fontsize=13,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1,
    )

    # Optimize layout for publication quality
    plt.tight_layout()

    # Adjust layout to make room for model labels and figure legend
    if num_models > 1:
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0, top=0.92, wspace=0.15, hspace=-0.1)
    else:
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0, top=0.92, wspace=0.15)

    # Save the plot
    output_path = output_dir / output_filename
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    print(f"Comparison polar plots saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create comparison polar plots for multiple models"
    )
    parser.add_argument(
        "--model-files",
        type=str,
        nargs="+",
        required=True,
        help="List of model_name:file_path pairs (e.g., 'GPT-4V:/path/to/raw_evaluation.json')",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory to save the comparison plots")
    parser.add_argument(
        "--output-filename",
        type=str,
        default="comparison_polar_plots.pdf",
        help="Name of the output PDF file (default: comparison_polar_plots.pdf)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse model files
    model_files = {}
    for model_file_str in args.model_files:
        if ":" not in model_file_str:
            raise ValueError(
                f"Invalid format for model file: {model_file_str}. Expected 'model_name:file_path'"
            )

        model_name, file_path_str = model_file_str.split(":", 1)
        file_path = Path(file_path_str)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        model_files[model_name] = file_path

    if args.output_dir is None:
        output_dir = (STRIDEQA_BENCH_ROOT / "results/lsr_polar_plots").resolve()
    else:
        output_dir = Path(args.output_dir)

    print(f"Creating comparison plots for {len(model_files)} models:")
    for model_name, file_path in model_files.items():
        print(f"  - {model_name}: {file_path}")

    create_comparison_polar_plots(
        model_files=model_files, output_dir=output_dir, output_filename=args.output_filename
    )


if __name__ == "__main__":
    main()
