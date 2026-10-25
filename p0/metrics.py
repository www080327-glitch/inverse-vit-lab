"""Unclipped pixel metrics; target access is confined to evaluation."""
import numpy as np


def reconstruction(predictions, targets, masks):
    delta = predictions.astype(np.float64)-targets
    foreground = np.broadcast_to(masks[:, None].astype(bool), delta.shape)
    return {"mse": float(np.mean(delta**2)), "mae": float(np.mean(np.abs(delta))),
            "foreground_mse": float(np.mean(delta[foreground]**2)),
            "foreground_mae": float(np.mean(np.abs(delta[foreground]))),
            "background_mse": float(np.mean(delta[~foreground]**2)),
            "background_mae": float(np.mean(np.abs(delta[~foreground]))),
            "range": [float(predictions.min()), float(predictions.max())],
            "out_of_range_fraction": float(((predictions < 0) | (predictions > 1)).mean()),
            "mean_pixel_variance_across_conditions": float(np.var(predictions, axis=0).mean())}


def nearest_targets(outputs, targets):
    records = []
    for i, value in enumerate(outputs):
        distances = np.mean((targets.astype(np.float64)-value)**2, axis=(1, 2, 3))
        j = int(np.argmin(distances))
        records.append({"output_id": i, "nearest_train_id": j, "nearest_pixel_mse": float(distances[j])})
    return records
