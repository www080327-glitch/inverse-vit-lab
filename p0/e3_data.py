"""E3 offline coordinate targets; never imported by the inference program."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from runtime import sha256


def fixed_spec(path="configs/e3.json"):
    return json.loads(Path(path).read_text())


def normalized_coordinates(xy):
    return np.asarray(xy, dtype=np.float32)*np.float32(2/47)-np.float32(1)


def fixed_projection(seed=61023):
    return np.random.default_rng(seed).standard_normal((2, 64)).astype("float32")


def build_arrays(spec):
    axis = np.array(spec["grid_axis_pixels"], dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.stack([xx.ravel(), yy.ravel()], axis=1)
    ids = np.arange(64)
    interpolation_ids = np.array([r*8+c for r, c in spec["interpolation_row_col"]])
    extrapolation_ids = np.array([i for i in ids if i % 8 in spec["extrapolation_columns"]])
    heldout_ids = np.sort(np.r_[interpolation_ids, extrapolation_ids])
    train_ids = np.setdiff1d(ids, heldout_ids)
    projection = fixed_projection(spec["projection_seed"])
    normalized = normalized_coordinates(coordinates)
    conditions = normalized @ projection
    y, x = np.indices((48, 48))
    masks = np.stack([(x-cx)**2+(y-cy)**2 <= spec["radius_pixels"]**2 for cx, cy in coordinates])
    background = np.array(spec["background_rgb"], dtype=np.float32)/255
    foreground = np.array(spec["circle_rgb"], dtype=np.float32)/255
    images = np.where(masks[:, None], foreground[None, :, None, None], background[None, :, None, None]).astype("float32")
    return {"images": images, "masks": masks, "conditions": conditions, "coordinates": coordinates,
            "normalized_coordinates": normalized, "ids": ids, "projection": projection,
            "train_ids": train_ids, "heldout_ids": heldout_ids,
            "interpolation_ids": interpolation_ids, "extrapolation_ids": extrapolation_ids}


def create_e3(out, spec_path="configs/e3.json"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    spec = fixed_spec(spec_path)
    arrays = build_arrays(spec)
    train = arrays["train_ids"]
    (out / "train").mkdir()
    # Original train.py reads ONLY this physical 48-example file.
    np.savez(out / "train" / "dataset.npz", **{k: arrays[k][train] for k in ["images", "masks", "conditions", "coordinates", "ids"]})
    np.savez(out / "all.npz", **arrays)
    np.save(out / "heldout_conditions.npy", arrays["conditions"][arrays["heldout_ids"]])
    targets = out / "targets"
    targets.mkdir()
    records = []
    for i in arrays["ids"]:
        i = int(i)
        label = "train" if i in train else ("interpolation" if i in arrays["interpolation_ids"] else "extrapolation")
        image = Image.fromarray((arrays["images"][i].transpose(1, 2, 0)*255).round().astype("uint8"))
        image.save(targets / f"{i:03d}.png")
        Image.fromarray(arrays["masks"][i].astype("uint8")*255).save(targets / f"{i:03d}_mask.png")
        records.append({"id": i, "row": i//8, "col": i%8, "xy_pixels": arrays["coordinates"][i].tolist(),
                        "normalized_xy": arrays["normalized_coordinates"][i].tolist(), "split": label})
    manifest = {"spec": spec, "records": records, "train_ids": train.tolist(),
                "heldout_ids": arrays["heldout_ids"].tolist(), "interpolation_ids": arrays["interpolation_ids"].tolist(),
                "extrapolation_ids": arrays["extrapolation_ids"].tolist(),
                "training_coordinate_convex_hull": [[9, 9], [33, 9], [33, 37], [9, 37]],
                "projection_matrix": arrays["projection"].tolist(),
                "train_dataset_sha256": sha256(out / "train" / "dataset.npz"), "all_dataset_sha256": sha256(out / "all.npz"),
                "foreground_pixels_per_image": int(arrays["masks"][0].sum()), "display_order": arrays["heldout_ids"].tolist()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Pre-training, complete split visualization. Rendering has no inference role.
    canvas = Image.new("RGB", (8*104, 8*124), "white")
    draw = ImageDraw.Draw(canvas)
    for row in records:
        i = row["id"]
        x, y = i % 8*104, i//8*124
        image = Image.open(targets / f"{i:03d}.png").resize((96, 96), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y+24))
        draw.text((x, y), f"{i} {row['split'][:6]}", fill="black")
    canvas.save(out / "split_grid.png")
    return manifest


def load_e3(path):
    with np.load(Path(path) / "all.npz", allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/e3")
    p.add_argument("--spec", default="configs/e3.json")
    a = p.parse_args()
    manifest = create_e3(a.out, a.spec)
    print(json.dumps({k: manifest[k] for k in ["train_ids", "heldout_ids", "interpolation_ids", "extrapolation_ids", "train_dataset_sha256", "foreground_pixels_per_image"]}, indent=2))
