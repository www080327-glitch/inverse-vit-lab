"""Fixed offline centroid/area/shape metrics. These never modify predictions."""
import numpy as np
from metrics import reconstruction


def foreground_mask(output, background, foreground, threshold=.5):
    background = np.asarray(background, dtype=np.float64)/255
    difference = np.asarray(foreground, dtype=np.float64)/255-background
    score = np.sum((output-background[:, None, None])*difference[:, None, None], axis=0)/np.sum(difference**2)
    return score >= threshold


def translate_zero(mask, dx, dy):
    translated = np.zeros_like(mask)
    h, w = mask.shape
    x0, x1, y0, y1 = max(0, -dx), min(w, w-dx), max(0, -dy), min(h, h-dy)
    if x0 < x1 and y0 < y1:
        translated[y0+dy:y1+dy, x0+dx:x1+dx] = mask[y0:y1, x0:x1]
    return translated


def iou(left, right):
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum()/union) if union else 1.


def geometry(output, target_mask, target_xy, spec):
    predicted = foreground_mask(output, spec["background_rgb"], spec["circle_rgb"], spec["foreground_threshold"])
    area = int(predicted.sum())
    target_area = int(target_mask.sum())
    if area:
        y, x = np.nonzero(predicted)
        center = np.array([x.mean(), y.mean()])
        center_error = float(np.linalg.norm(center-target_xy))
        shift = np.rint(np.asarray(target_xy)-center).astype(int)
        aligned = translate_zero(predicted, int(shift[0]), int(shift[1]))
        aligned_iou = iou(aligned, target_mask)
        xy = np.stack([x, y], axis=1)-center
        eigenvalues = np.linalg.eigvalsh(xy.T @ xy/area)
        anisotropy = float((eigenvalues[-1]-eigenvalues[0])/max(eigenvalues.sum(), 1e-12))
        center_value = center.tolist()
    else:
        center_value, center_error, aligned_iou, anisotropy = None, 47*2**.5, 0., None
    return {"predicted_center_xy": center_value, "centroid_valid": bool(area), "center_error_pixels": center_error,
            "predicted_area": area, "target_area": target_area, "area_signed_error_pixels": area-target_area,
            "area_absolute_error_pixels": abs(area-target_area), "area_relative_absolute_error": abs(area-target_area)/target_area,
            "mask_iou": iou(predicted, target_mask), "aligned_mask_iou": aligned_iou,
            "aligned_shape_error": 1-aligned_iou, "anisotropy": anisotropy}


def summarize(outputs, targets, masks, coordinates, ids, spec):
    per_sample = []
    for i, output, target, mask, coordinate in zip(ids, outputs, targets, masks, coordinates):
        metric = reconstruction(output[None], target[None], mask[None])
        per_sample.append({"id": int(i), "mse": metric["mse"], "foreground_mse": metric["foreground_mse"],
                           **geometry(output, mask, coordinate, spec)})
    result = reconstruction(outputs, targets, masks)
    for key in ["center_error_pixels", "area_absolute_error_pixels", "area_relative_absolute_error", "mask_iou", "aligned_mask_iou", "aligned_shape_error"]:
        result[key] = float(np.mean([r[key] for r in per_sample]))
    result["center_error_pixels_max"] = max(r["center_error_pixels"] for r in per_sample)
    result["invalid_centroids"] = sum(not r["centroid_valid"] for r in per_sample)
    result["count"] = len(per_sample)
    result["per_sample"] = per_sample
    return result


def nearest_coordinate_ids(coordinates, training_ids):
    # Sorted candidates make numpy argmin ties choose the lowest global ID.
    training_ids = np.sort(training_ids)
    distances = ((coordinates[:, None]-coordinates[training_ids][None])**2).sum(axis=2)
    return training_ids[np.argmin(distances, axis=1)]


def compare_to_baselines(model, baselines, spec):
    keys = ["mse", "foreground_mse", "center_error_pixels", "mask_iou"]
    comparisons = {name: {key: {"model": model[key], "baseline": baseline[key],
                               "strictly_better": bool(model[key] > baseline[key] if key == "mask_iou" else model[key] < baseline[key])} for key in keys}
                   for name, baseline in baselines.items()}
    thresholds = spec["quality_thresholds"]
    quality = {"center": model["center_error_pixels"] <= thresholds["mean_centroid_error_pixels_max"],
               "area": model["area_relative_absolute_error"] <= thresholds["mean_relative_area_error_max"],
               "shape": model["aligned_shape_error"] <= thresholds["mean_aligned_shape_error_max"],
               "iou": model["mask_iou"] >= thresholds["mean_mask_iou_min"],
               "all_centroids_valid": model["invalid_centroids"] == 0}
    passed = all(quality.values()) and all(value["strictly_better"] for baseline in comparisons.values() for value in baseline.values())
    return {"status": "PASS_SCOPED" if passed else "FAIL", "comparisons": comparisons, "quality_checks": quality}
