from numbers import Real

import numpy as np


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages."""

    def __init__(self):
        self._dict = dict(
            a1=RunningAverage(),
            a2=RunningAverage(),
            a3=RunningAverage(),
            abs_rel=RunningAverage(),
            rmse=RunningAverage(),
            log_10=RunningAverage(),
            rmse_log=RunningAverage(),
            silog=RunningAverage(),
            sq_rel=RunningAverage(),
        )

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key in new_dict.keys():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Convert inputs to float, filter out invalid values, and compute depth estimation metrics.

    Args:
        gt (np.ndarray): Ground truth array. shape (N, )
        pred (np.ndarray): Prediction array. shape (N, )

    Returns:
        dict: Dictionary containing various error metrics (a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel).
              If no valid values remain after filtering, returns an empty dict.
    """
    # Force cast to float. Invalid parsing -> NaN
    gt = np.array(gt, dtype=float)
    pred = np.array(pred, dtype=float)

    # Replace zero or negative values with a small positive epsilon for logs
    epsilon = 1e-8
    gt = np.where(gt <= 0, epsilon, gt)
    pred = np.where(pred <= 0, epsilon, pred)

    # Filter out any NaNs or Infs introduced by casting, etc.
    mask = np.isfinite(gt) & np.isfinite(pred)
    if not np.any(mask):
        return {}
    gt = gt[mask]
    pred = pred[mask]

    # Compute metrics
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - (np.mean(err)) ** 2) * 100

    log_10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "silog": silog,
        "log_10": log_10,
    }


def judge_success(
    qa_type: str,
    gt_value: Real,
    pred_value: Real | None,
    tolerance: dict,
) -> dict:
    def relative(pred: Real, gt: Real, tolerance_config: dict) -> dict:
        if not isinstance(pred, Real):
            return {"success": False, "reason": "invalid_prediction"}

        relative_range = tolerance_config["range"]

        # Check if a hybrid absolute tolerance floor is defined and the GT is below the threshold
        if (
            "absolute_floor" in tolerance_config
            and gt < tolerance_config["absolute_floor"]["threshold"]
        ):
            # Use a fixed absolute tolerance calculated AT the threshold
            threshold = tolerance_config["absolute_floor"]["threshold"]
            abs_tolerance = threshold * relative_range  # Dynamic calculation

            success = abs(pred - gt) <= abs_tolerance
            min_value = gt - abs_tolerance
            max_value = gt + abs_tolerance
            fn_used = "absolute_floor"
        else:
            # Default to relative tolerance for values above the threshold
            min_value = gt * (1 - relative_range)
            max_value = gt * (1 + relative_range)
            success = min_value <= pred <= max_value
            fn_used = "relative"

        return {
            "success": success,
            "min_value": min_value,
            "max_value": max_value,
            "pred": pred,
            "gt": gt,
            "fn_used": fn_used,
        }

    def absolute(pred: Real, gt: Real, tolerance_config: dict) -> dict:
        if not isinstance(pred, Real):
            return {"success": False, "reason": "invalid_prediction"}

        tolerance_range = tolerance_config["range"]
        success = abs(pred - gt) <= tolerance_range
        return {
            "success": success,
            "tolerance_range": tolerance_range,
            "diff": abs(pred - gt),
            "pred": pred,
            "gt": gt,
        }

    def angular_absolute(pred: Real, gt: Real, tolerance_config: dict) -> dict:
        if not isinstance(pred, Real):
            return {"success": False, "reason": "invalid_prediction"}

        tolerance_range = tolerance_config["range"]
        # Calculate angular difference considering wraparound (-180 to 180)
        diff = (pred - gt + 180) % 360 - 180
        success = abs(diff) <= tolerance_range
        return {
            "success": success,
            "tolerance_range": tolerance_range,
            "diff": abs(diff),
            "pred": pred,
            "gt": gt,
        }

    def exact_match(pred: Real, gt: Real, tolerance_config: dict) -> dict:
        if not isinstance(pred, Real):
            return {"success": False, "reason": "invalid_prediction"}

        success = pred == gt
        return {
            "success": success,
            "pred": pred,
            "gt": gt,
        }

    judge_functions = {
        "absolute": absolute,
        "relative": relative,
        "exact": exact_match,
        "angular_absolute": angular_absolute,
    }

    judge_fn_name = tolerance.get("fn")
    if judge_fn_name not in judge_functions:
        raise ValueError(f"Unsupported judge function: {judge_fn_name}")

    return judge_functions[judge_fn_name](pred_value, gt_value, tolerance)


def judge_localization_success(
    distance_gt: Real,
    direction_gt: Real,
    distance_pred: Real | None,
    direction_pred: Real | None,
    tolerance_config: dict,
) -> dict:
    """
    Judge localization success for polar coordinate position evaluation.
    This function evaluates whether a predicted position falls within the acceptable region.

    Args:
        distance_gt (Real): Ground truth distance in meters
        direction_gt (Real): Ground truth direction in degrees (-180 to 180)
        distance_pred (Real | None): Predicted distance in meters
        direction_pred (Real | None): Predicted direction in degrees
        tolerance_config (dict): Configuration containing tolerance settings for distance and direction

    Returns:
        dict: Dictionary containing:
            - success (bool): Whether both distance and direction are within tolerance
            - distance_success (bool): Whether distance is within tolerance
            - direction_success (bool): Whether direction is within tolerance
            - distance_details (dict): Detailed distance evaluation results
            - direction_details (dict): Detailed direction evaluation results

    """
    # Get tolerance configurations for each component
    distance_tolerance = tolerance_config.get("distance", {})
    direction_tolerance = tolerance_config.get("direction", {})

    # Evaluate distance and direction by calling the generic judge_success function
    distance_details = judge_success("distance", distance_gt, distance_pred, distance_tolerance)
    direction_details = judge_success(
        "direction", direction_gt, direction_pred, direction_tolerance
    )

    distance_success = distance_details.get("success", False)
    direction_success = direction_details.get("success", False)

    # LSR success requires both to be within tolerance
    lsr_success = distance_success and direction_success

    return {
        "success": lsr_success,
        "distance_success": distance_success,
        "direction_success": direction_success,
        "distance_details": distance_details,
        "direction_details": direction_details,
    }


def _evaluate_distance_tolerance(gt: Real, pred: Real, tolerance: dict) -> dict:
    """Helper function to evaluate distance tolerance."""
    tolerance_range = tolerance["range"]

    min_value = gt * (1 - tolerance_range)
    max_value = gt * (1 + tolerance_range)
    success = min_value <= pred <= max_value

    return {
        "success": success,
        "min_value": min_value,
        "max_value": max_value,
        "pred": pred,
        "gt": gt,
        "tolerance_range": tolerance_range,
        "fn": "relative",
    }


def _evaluate_direction_tolerance(gt: Real, pred: Real, tolerance: dict) -> dict:
    """Helper function to evaluate direction tolerance with angular wraparound."""
    tolerance_range = tolerance["range"]

    # Calculate angular difference considering wraparound (-180 to 180)
    diff = (pred - gt + 180) % 360 - 180
    success = abs(diff) <= tolerance_range

    return {
        "success": success,
        "tolerance_range": tolerance_range,
        "angular_diff": abs(diff),
        "pred": pred,
        "gt": gt,
        "fn": "angular_absolute",
    }
