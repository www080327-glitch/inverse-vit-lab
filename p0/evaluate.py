"""Full ordered reconstruction, condition controls, unseen/interpolation diagnostics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data import load_dataset
from metrics import nearest_targets, reconstruction
from runtime import load_model, sha256
from sample import infer, save_images
from visuals import comparisons, diagnostic_grid


def evaluate(run, data, device):
    run, data = Path(run), Path(data)
    raw = load_dataset(data)
    plan = json.loads((data / "manifest.json").read_text())
    model = load_model(run / "weights.pt", device)
    after = infer(model, raw["conditions"])
    saved_after = np.load(run / "after" / "outputs.npy")
    reload_max_error = float(np.abs(after-saved_after).max())
    if not np.allclose(after, saved_after, atol=1e-5, rtol=1e-5):
        raise AssertionError("Reload output mismatch")
    before = np.load(run / "before" / "outputs.npy")
    targets, masks = raw["images"], raw["masks"]
    baseline = np.broadcast_to(targets.mean(axis=0), targets.shape)
    # Fixed derangement, no unchanged target-condition pairs.
    permutation = np.roll(np.arange(len(targets)), 1)
    shuffled = infer(model, raw["conditions"][permutation])
    mean_condition = infer(model, np.repeat(raw["conditions"].mean(axis=0, keepdims=True), len(targets), axis=0))
    result = {"parameters": model.parameter_count(), "count": len(targets), "device": device,
              "before": reconstruction(before, targets, masks), "after": reconstruction(after, targets, masks),
              "train_mean_image_baseline": reconstruction(baseline, targets, masks),
              "shuffled_conditions": reconstruction(shuffled, targets, masks),
              "mean_condition_control": reconstruction(mean_condition, targets, masks),
              "shuffle_permutation": permutation.tolist(), "reload_max_abs_error": reload_max_error,
              "reload_tolerance": {"atol": 1e-5, "rtol": 1e-5},
              "per_sample": [{"id": i, "mse": float(np.mean((after[i]-targets[i])**2)),
                              "mae": float(np.mean(np.abs(after[i]-targets[i])))} for i in range(len(targets))],
              "weights_sha256": sha256(run / "weights.pt"), "data_sha256": sha256(data / "dataset.npz")}
    comparisons(targets, before, after, run / "comparisons")
    diag = run / "diagnostics"
    diag.mkdir(exist_ok=True)
    unseen = np.random.default_rng(plan["unseen_seed"]).standard_normal((plan["unseen_count"], 64)).astype("float32")
    np.save(diag / "unseen_conditions.npy", unseen)
    outputs = infer(model, unseen)
    save_images(outputs, diag / "unseen")
    diagnostic_grid(outputs, [f"unseen {i}" for i in range(len(outputs))], 4, diag / "unseen_grid.png")
    result["unseen"] = {"no_unique_ground_truth": True, "seed": plan["unseen_seed"],
                         "condition_norms": np.linalg.norm(unseen, axis=1).tolist(),
                         "nearest_train_targets": nearest_targets(outputs, targets),
                         "output_range": [float(outputs.min()), float(outputs.max())],
                         "out_of_range_fraction": float(((outputs < 0) | (outputs > 1)).mean()),
                         "mean_distance_to_train_mean": float(np.mean((outputs-targets.mean(axis=0))**2)),
                         "output_variance": float(np.var(outputs, axis=0).mean())}
    vectors, labels, records = [], [], []
    for left, right in plan["interpolation_pairs"]:
        for alpha in plan["alphas"]:
            vector = ((1-alpha)*raw["conditions"][left]+alpha*raw["conditions"][right]).astype("float32")
            vectors.append(vector)
            labels.append(f"{left}->{right} a={alpha}")
            records.append({"left": left, "right": right, "alpha": alpha, "condition_norm": float(np.linalg.norm(vector))})
    vectors = np.stack(vectors)
    np.save(diag / "interpolation_conditions.npy", vectors)
    outputs = infer(model, vectors)
    save_images(outputs, diag / "interpolation")
    diagnostic_grid(outputs, labels, 5, diag / "interpolation_grid.png")
    nearest = nearest_targets(outputs, targets)
    for i, record in enumerate(records):
        record.update(nearest[i])
        if record["alpha"] in (0, 1):
            endpoint = record["left"] if record["alpha"] == 0 else record["right"]
            record["endpoint_max_abs_error"] = float(np.abs(outputs[i]-after[endpoint]).max())
    result["interpolation"] = {"no_unique_ground_truth_for_interior": True, "records": records}
    (run / "metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    rows = [json.loads(line) for line in (run / "loss.jsonl").read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in rows], [r["batch_mse_before_update"] for r in rows], label="batch before update")
    full = [r for r in rows if "full_train_mse_after_update" in r]
    ax.plot([r["step"] for r in full], [r["full_train_mse_after_update"] for r in full], label="full train after update")
    ax.axhline(result["train_mean_image_baseline"]["mse"], color="gray", linestyle="--", label="train mean baseline")
    ax.set(xlabel="optimizer steps", ylabel="unclipped MSE", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ["parameters", "count", "before", "after", "train_mean_image_baseline", "shuffled_conditions", "reload_max_abs_error"]}, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", choices=["cpu", "mps"], default="mps")
    a = p.parse_args()
    evaluate(a.run, a.data, a.device)
