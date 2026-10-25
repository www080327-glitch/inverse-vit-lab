"""E3 evaluation only, after last-step training; no checkpoint selection."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from e3_data import load_e3
from e3_metrics import compare_to_baselines, nearest_coordinate_ids, summarize
from runtime import load_model, sha256
from sample import infer, save_images
from visuals import rgb


def render_heldout(arrays, ids, before, after, nearest, mean, run):
    folder = run / "heldout_comparisons"
    folder.mkdir(exist_ok=False)
    labels = ["TARGET", "BEFORE", "MODEL LAST", "NEAREST XY", "TRAIN MEAN"]
    for start in range(0, len(ids), 8):
        selected = ids[start:start+8]
        canvas = Image.new("RGB", (650, 30+len(selected)*110), "white")
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(labels):
            draw.text((95+col*110, 5), label, fill="black")
        for row, i in enumerate(selected):
            label = "interp" if i in arrays["interpolation_ids"] else "extrap"
            draw.text((2, 40+row*110), f"ID {i}\n{label}", fill="black")
            for col, value in enumerate([arrays["images"][i], before[i], after[i], nearest[i], mean[i]]):
                canvas.paste(rgb(value, scale=2), (95+col*110, 30+row*110))
        canvas.save(folder / f"heldout_page_{start//8+1:02d}.png")


def evaluate_e3(run, data="data/e3", device="mps"):
    run, data = Path(run), Path(data)
    arrays = load_e3(data)
    manifest = json.loads((data / "manifest.json").read_text())
    spec = manifest["spec"]
    result_path = run / "e3_metrics.json"
    if result_path.exists():
        raise FileExistsError("E3 evaluation refuses to overwrite an existing report")
    model = load_model(run / "weights.pt", device)
    predictions = infer(model, arrays["conditions"])
    initial = infer(load_model(run / "initial_weights.pt", device), arrays["conditions"])
    train_ids, heldout_ids = arrays["train_ids"], arrays["heldout_ids"]
    expected_train = np.load(run / "after" / "outputs.npy", allow_pickle=False)
    max_error = float(np.abs(predictions[train_ids]-expected_train).max())
    np.testing.assert_allclose(predictions[train_ids], expected_train, atol=1e-5, rtol=1e-5)
    mean_images = np.broadcast_to(arrays["images"][train_ids].mean(axis=0), arrays["images"].shape)
    nearest_ids = nearest_coordinate_ids(arrays["coordinates"], train_ids)
    nearest_images = arrays["images"][nearest_ids]
    result = {"seed": json.loads((run / "invocation_000000.json").read_text())["seed"],
              "parameters": model.parameter_count(), "device": device, "dtype": "float32",
              "weights_sha256": sha256(run / "weights.pt"), "data_sha256": sha256(data / "all.npz"),
              "train_reload_max_abs_error": max_error, "last_checkpoint_only": True,
              "heldout_access_during_training": False, "nearest_train_id_by_global_id": nearest_ids.tolist(), "groups": {}}
    for group, ids in {"train": train_ids, "heldout": heldout_ids, "interpolation": arrays["interpolation_ids"], "extrapolation": arrays["extrapolation_ids"]}.items():
        args = [arrays["images"][ids], arrays["masks"][ids], arrays["coordinates"][ids], ids, spec]
        metrics = summarize(predictions[ids], *args)
        baselines = {"train_mean": summarize(mean_images[ids], *args), "nearest_coordinate": summarize(nearest_images[ids], *args)}
        result["groups"][group] = {"model": metrics, "before": summarize(initial[ids], *args), "baselines": baselines,
                                    "gate": compare_to_baselines(metrics, baselines, spec) if group != "train" else {"status": "NOT_APPLICABLE_TRAIN_SELF_NEIGHBOR"}}
    for name, values in [("heldout_targets", arrays["images"][heldout_ids]), ("heldout_before", initial[heldout_ids]),
                         ("heldout_after", predictions[heldout_ids]), ("heldout_nearest_baseline", nearest_images[heldout_ids]),
                         ("heldout_mean_baseline", mean_images[heldout_ids])]:
        save_images(values, run / name)
    (run / "heldout_display_ids.json").write_text(json.dumps(heldout_ids.tolist())+"\n")
    render_heldout(arrays, heldout_ids, initial, predictions, nearest_images, mean_images, run)
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    rows = [json.loads(line) for line in (run / "loss.jsonl").read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in rows], [r["batch_mse_before_update"] for r in rows], label="train batch")
    full = [r for r in rows if "full_train_mse_after_update" in r]
    ax.plot([r["step"] for r in full], [r["full_train_mse_after_update"] for r in full], label="full train")
    ax.set(xlabel="optimizer steps", ylabel="unclipped MSE", yscale="log", title=f"E3 seed {result['seed']} (no holdout curve)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "e3_loss_curve.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"seed": result["seed"], "reload_max_abs_error": max_error,
                      "groups": {k: {"mse": v["model"]["mse"], "foreground_mse": v["model"]["foreground_mse"],
                                      "center_error_pixels": v["model"]["center_error_pixels"], "gate": v["gate"]["status"]} for k, v in result["groups"].items()}}, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="data/e3")
    a = p.parse_args()
    evaluate_e3(a.run, a.data)
