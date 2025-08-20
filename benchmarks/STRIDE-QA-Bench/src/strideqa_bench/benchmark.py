import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tabulate import tabulate
from tqdm import tqdm

from strideqa_bench import STRIDEQA_BENCH_ROOT
from strideqa_bench.utils.evaluate_utils import save_as_jsonl
from strideqa_bench.utils.metric_utils import judge_localization_success, judge_success
from strideqa_bench.utils.parser import parse_pred_value
from strideqa_bench.visualize.visualize import (
    create_lsr_analysis_plots,
    create_prediction_distribution_plots,
)

# qa_category -> time
TIME_BUCKET: dict[str, int] = {
    # t = 0
    "ego_distance_data": 0,
    "ego_speed_data": 0,
    "target_speed_data": 0,
    "target_bearing_angle_data": 0,
    # t = 1
    "ego_distance_data_4d_t1": 1,
    "ego_speed_data_4d_t1": 1,
    "target_speed_data_4d_t1": 1,
    # t = 2
    "ego_distance_data_4d_t2": 2,
    "ego_speed_data_4d_t2": 2,
    "target_speed_data_4d_t2": 2,
    # t = 3
    "ego_distance_data_4d_t3": 3,
    "ego_speed_data_4d_t3": 3,
    "target_speed_data_4d_t3": 3,
}

# qa_category -> ego or target
EGO_TARGET_DISTINCTION: dict[str, str] = {
    # t = 0
    "ego_distance_data": "target",
    "ego_speed_data": "ego",
    "target_speed_data": "target",
    "target_bearing_angle_data": "target",
    # t = 1
    "ego_distance_data_4d_t1": "target",
    "ego_speed_data_4d_t1": "ego",
    "target_speed_data_4d_t1": "target",
    # t = 2
    "ego_distance_data_4d_t2": "target",
    "ego_speed_data_4d_t2": "ego",
    "target_speed_data_4d_t2": "target",
    # t = 3
    "ego_distance_data_4d_t3": "target",
    "ego_speed_data_4d_t3": "ego",
    "target_speed_data_4d_t3": "target",
}


def extract_group_id(question_id: str) -> str:
    """Extract group ID from question_id (e.g., 'q00000234_g00018' -> 'g00018')"""
    if "_g" in question_id:
        return "g" + question_id.split("_g")[1]
    return question_id  # fallback to original if format doesn't match


def load_annotation_data(annotation_csv_path: Path | str) -> dict[str, dict[str, str]]:
    """Load annotation data from CSV file and return a dictionary mapping group_id to tags."""
    df = pd.read_csv(annotation_csv_path)
    annotation_dict = {}

    for _, row in df.iterrows():
        group_id = row["group_id"]
        object_type = row["object_type"] if pd.notna(row["object_type"]) else None
        relation = row["relation"] if pd.notna(row["relation"]) else None

        annotation_dict[group_id] = {"object_type": object_type, "relation": relation}

    return annotation_dict


def run_evaluation(
    model_response_list: list[dict[str, Any]],
    config: dict[str, Any],
    annotation_dict: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    raw_results_list: list[dict[str, Any]] = []

    for model_response in tqdm(model_response_list):
        gt_value_dict = model_response.get("gt_value", {})

        time_subscript = TIME_BUCKET[model_response["qa_category"]]
        ego_or_target = EGO_TARGET_DISTINCTION[model_response["qa_category"]]

        # Extract group_id and get annotation information
        question_id = model_response.get("question_id", "")
        group_id = extract_group_id(question_id)
        annotation_info = annotation_dict.get(group_id, {})
        object_type = annotation_info.get("object_type")
        relation = annotation_info.get("relation")

        for qa_type, gt_value in gt_value_dict.items():
            gt_unit = model_response["unit"][qa_type]
            tolerance = config["tolerance"][qa_type]

            pred: str = model_response.get("pred", "")
            parsed_result = parse_pred_value(qa_type, pred)
            pred_value, pred_unit = parsed_result if parsed_result is not None else (None, None)

            details = judge_success(qa_type, gt_value, pred_value, tolerance)
            success = details["success"]
            details.update(tolerance)

            raw_results = {
                "question_id": model_response.get("question_id"),
                "group_id": group_id,
                "object_type": object_type,
                "relation": relation,
                "qa_category": model_response["qa_category"],
                "time": time_subscript,
                "ego_or_target": ego_or_target,
                "qa_type": qa_type,
                "gt": model_response.get("gt"),
                "pred": pred,
                "gt_value": gt_value,
                "pred_value": pred_value,
                "gt_unit": gt_unit,
                "pred_unit": pred_unit,
                "success": success,
                "details": details,
            }

            raw_results_list.append(raw_results)

    # Calculate LSR for pairs of distance and direction data
    lsr_results_list = _calculate_localization_success(raw_results_list, config, annotation_dict)
    raw_results_list.extend(lsr_results_list)

    return raw_results_list


def _calculate_localization_success(
    raw_results_list: list[dict[str, Any]],
    config: dict[str, Any],
    annotation_dict: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Calculate Localization Success Rate (LSR) for distance-direction pairs.

    Args:
        raw_results_list: List of individual evaluation results
        config: Configuration dictionary containing tolerance settings

    Returns:
        List of LSR evaluation results
    """
    lsr_results: list[dict[str, Any]] = []

    # Group results by image_group, time, and ego_or_target
    # This allows pairing distance and direction questions from the same image group
    grouped_data: dict[str, dict] = {}

    for result in raw_results_list:
        question_id = result["question_id"]
        image_group = extract_group_id(question_id)
        time = result["time"]
        ego_or_target = result["ego_or_target"]
        qa_type = result["qa_type"]

        # Create a unique key for grouping based on image_group, time, and ego_or_target
        # This allows pairing distance and direction from the same image group
        group_key = f"{image_group}_{time}_{ego_or_target}"

        if group_key not in grouped_data:
            grouped_data[group_key] = {
                "image_group": image_group,
                "time": time,
                "ego_or_target": ego_or_target,
                "distance": None,
                "direction": None,
            }

        if qa_type == "distance":
            grouped_data[group_key]["distance"] = result
        elif qa_type == "direction":
            grouped_data[group_key]["direction"] = result

    # Calculate LSR for groups - now handle cases where either distance or direction is missing
    for group_key, group_data in grouped_data.items():
        distance_result = group_data["distance"]
        direction_result = group_data["direction"]

        # Skip only if both distance and direction are missing (no data for this group)
        if distance_result is None and direction_result is None:
            continue

        # Determine which result to use as primary source for metadata
        primary_result = distance_result if distance_result is not None else direction_result

        # Extract or set default values for LSR calculation
        distance_gt = distance_result["gt_value"] if distance_result is not None else None
        direction_gt = direction_result["gt_value"] if direction_result is not None else None
        distance_pred = distance_result["pred_value"] if distance_result is not None else None
        direction_pred = direction_result["pred_value"] if direction_result is not None else None

        # Calculate LSR using the metric_utils function
        lsr_details = judge_localization_success(
            distance_gt=distance_gt,
            direction_gt=direction_gt,
            distance_pred=distance_pred,
            direction_pred=direction_pred,
            tolerance_config=config["tolerance"],
        )

        # Use the qa_category from primary result for LSR
        primary_qa_category = primary_result["qa_category"]

        # Get annotation information from primary result
        group_id = primary_result.get("group_id", "")
        object_type = primary_result.get("object_type")
        relation = primary_result.get("relation")

        # Create LSR result entry using primary question_id
        lsr_result = {
            "question_id": primary_result["question_id"],  # Use primary question_id
            "group_id": group_id,
            "object_type": object_type,
            "relation": relation,
            "qa_category": primary_qa_category,
            "time": group_data["time"],
            "ego_or_target": group_data["ego_or_target"],
            "qa_type": "lsr",
            "gt_value": {"distance": distance_gt, "direction": direction_gt},
            "pred_value": {"distance": distance_pred, "direction": direction_pred},
            "gt_unit": {
                "distance": distance_result["gt_unit"] if distance_result is not None else None,
                "direction": direction_result["gt_unit"] if direction_result is not None else None,
            },
            "pred_unit": {
                "distance": distance_result["pred_unit"] if distance_result is not None else None,
                "direction": direction_result["pred_unit"]
                if direction_result is not None
                else None,
            },
            "success": lsr_details["success"],
            "details": lsr_details,
        }

        lsr_results.append(lsr_result)

    return lsr_results


def calculate_success_rate(
    raw_results_list: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    stats: dict[str, dict[int, list[int]]] = {}

    for raw_results in raw_results_list:
        qa_type = raw_results["qa_type"]
        subject = raw_results["ego_or_target"]
        time = raw_results["time"]
        success = raw_results["success"]

        # For LSR, aggregate under a unified key without subject prefix
        if qa_type.lower() == "lsr":
            metric_name = "lsr"
        else:
            metric_name = f"{subject.lower()}_{qa_type.lower()}"

        stats.setdefault(metric_name, {})
        stats[metric_name].setdefault(time, [0, 0])

        stats[metric_name][time][0] += 1
        if success:
            stats[metric_name][time][1] += 1

    results: dict[str, list[dict[str, Any]]] = {}
    for qa_type, time_stats in stats.items():
        results[qa_type] = []
        total_counts = 0
        total_successes = 0

        for time, (counts, successes) in sorted(time_stats.items()):
            rate = successes / counts if counts > 0 else 0
            results[qa_type].append(
                {"time": time, "success_rate": round(rate, 8), "sample_size": counts}
            )
            total_counts += counts
            total_successes += successes

        if total_counts > 0:
            avg_rate = total_successes / total_counts
            results[qa_type].append(
                {
                    "time": "avg",
                    "success_rate": round(avg_rate, 8),
                    "sample_size": total_counts,
                }
            )

    return results


def compute_tlc_mlsr(
    df: pd.DataFrame,
    *,
    max_step: int = 3,
) -> pd.DataFrame:
    """Compute TLC@k (k=1..max_step), TLC, MLSR

    Return DataFrame columns:
        relation | TC@1 | ... | TLC@max_step | TLC | MLSR

    - TLC@k : The proportion of cases where success is achieved at t=0 and then consecutively maintained for frames 1 through k.
    - TLC   : The proportion of cases where success is consecutively maintained across all frames (Temporal Localization Consistency).
    - MLSR  : For each sequence, compute "successful frames / (max_step + 1)" and then take the average per relation.

    """
    lsr = df[df.qa_type == "lsr"].copy()

    pivot = (
        lsr.pivot_table(
            index=["group_id", "relation"], columns="time", values="success", aggfunc="first"
        )
        .fillna(False)
        .astype(bool)
    )

    frames = []
    for k in range(1, max_step + 1):
        seq_ok = pivot[0] & pivot.loc[:, list(range(1, k + 1))].all(axis=1)
        rel_df = (
            seq_ok.reset_index(name="ok")
            .groupby("relation")["ok"]
            .mean()
            .reset_index(name=f"TLC@{k}")
        )
        overall = pd.DataFrame([{"relation": "Overall", f"TLC@{k}": seq_ok.mean()}])
        frames.append(pd.concat([overall, rel_df], ignore_index=True))

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="relation", how="outer")

    strict_ok = pivot.loc[:, list(range(0, max_step + 1))].all(axis=1)
    strict_df = (
        strict_ok.reset_index(name="ok").groupby("relation")["ok"].mean().reset_index(name="TLC")
    )
    strict_overall = pd.DataFrame([{"relation": "Overall", "TLC": strict_ok.mean()}])
    strict_df = pd.concat([strict_overall, strict_df], ignore_index=True)
    merged = merged.merge(strict_df, on="relation", how="outer")

    mf_score = pivot.mean(axis=1)
    mf_df = (
        mf_score.reset_index(name="score")
        .groupby("relation")["score"]
        .mean()
        .reset_index(name="MLSR")
    )
    mf_overall = pd.DataFrame([{"relation": "Overall", "MLSR": mf_score.mean()}])
    mf_df = pd.concat([mf_overall, mf_df], ignore_index=True)
    merged = merged.merge(mf_df, on="relation", how="outer")

    merged = pd.concat(
        [
            merged.loc[merged.relation == "Overall"],
            merged.loc[merged.relation != "Overall"].sort_values("TLC", ascending=False),
        ]
    ).reset_index(drop=True)

    return merged


def aggregate_results(raw_results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the aggregated results in dictionary format.
    TLC / MLSR are also calculated here.
    """
    jst = timezone(timedelta(hours=9))
    evaluated_at = datetime.now(jst).isoformat()
    result_dict: dict[str, Any] = {"evaluated_at": evaluated_at}

    # Success rates (each frame independently)
    success_rates = calculate_success_rate(raw_results_list)
    result_dict.update(success_rates)

    # Temporal consistency metrics
    df_all = pd.DataFrame(raw_results_list)
    df_lsr = df_all[df_all.qa_type == "lsr"]
    df_tlc_mlsr = compute_tlc_mlsr(df_lsr)
    result_dict["temporal_consistency"] = df_tlc_mlsr.to_dict(orient="records")

    return result_dict


def round_half_up(value: float, ndigits: int = 0) -> float:
    """Round half up."""
    multiplier = 10**ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def generate_metric_tables(data: dict[str, Any], filepath: str) -> None:
    """
    Write each metric (success_rates, temporal_consistency, etc.) as a markdown table.
    """
    ordered_keys = [
        "target_distance",
        "target_direction",
        "ego_velocity",
        "target_velocity",
        "lsr",
        "temporal_consistency",
    ]
    all_keys = set(data.keys())
    remaining_keys = sorted(all_keys - set(ordered_keys))

    reordered = {k: data.get(k) for k in ordered_keys if k in data}
    reordered |= {k: data[k] for k in remaining_keys}

    # Note: Keep this function deterministic for stable report diffs

    def fmt_val(v: Any) -> Any:
        if isinstance(v, float):
            return f"{round_half_up(v * 100, 1)}"
        return v

    with open(filepath, "w", encoding="utf-8") as f:
        for metric, records in reordered.items():
            if not isinstance(records, list) or not records:
                continue

            headers = list(records[0].keys())
            rows = [[fmt_val(rec[h]) for h in headers] for rec in records]

            f.write(f"### {metric.replace('_', ' ').title()}\n\n")
            f.write(tabulate(rows, headers, tablefmt="github"))
            f.write("\n\n")


def output_results(
    result_dict: dict[str, Any],
    raw_list: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)

    with (output_dir / "metrics_report.json").open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    print(f"Saved scores to {output_dir / 'metrics_report.json'}")

    generate_metric_tables(result_dict, output_dir / "metrics_report.md")
    print(f"Saved metric report to {output_dir / 'metrics_report.md'}")

    save_path = output_dir / "raw_evaluation.json"
    save_as_jsonl(raw_list, save_path, lines=False, ensure_ascii=False)
    print(f"Saved raw results to {save_path}")

    # Generate visualization plots
    print("Generating visualization plots...")
    create_prediction_distribution_plots(raw_list, output_dir)
    print(f"Saved visualization plots to {output_dir}")

    # Generate LSR analysis plots
    print("Generating LSR analysis plots...")
    create_lsr_analysis_plots(raw_list, output_dir)
    print(f"Saved LSR analysis plots to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLM responses on STRIDE-QA Bench.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing one or more *.json files with model responses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation results will be written.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="Annotation directory containing the annotation_sheet.csv.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=STRIDEQA_BENCH_ROOT / "config/tolerance.yaml",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load the config file
    with open(args.config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load annotation data from CSV
    annotation_csv_path = args.annotation_dir / "annotation_sheet.csv"

    if not annotation_csv_path.exists():
        print(f"Warning: Annotation CSV file not found at {annotation_csv_path}")
        print("Continuing without annotation information...")
        annotation_dict = {}
    else:
        print(f"Loading annotation data from {annotation_csv_path}")
        annotation_dict = load_annotation_data(annotation_csv_path)
        print(f"Loaded annotation data for {len(annotation_dict)} groups")

    # Load the json files
    model_response_list: list[dict] = []
    for json_file in sorted(args.input_dir.glob("*_t*.json")):
        with json_file.open("r", encoding="utf-8") as f_in:
            data = json.load(f_in)
            model_response_list.extend(data)

    if not model_response_list:
        raise ValueError(f"No model responses found in the provided directory: {args.input_dir}")

    # Run the evaluation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_results_list = run_evaluation(model_response_list, config, annotation_dict)

    # Aggregate the results
    result_dict = aggregate_results(raw_results_list)

    # Output the results
    output_results(result_dict, raw_results_list, args.output_dir)


if __name__ == "__main__":
    main(parse_args())
