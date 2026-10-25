"""Offline procedural targets. Never imported by sample.py or model.py."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def create_dataset(out, count, image_seed, condition_seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "targets").mkdir()
    rng = np.random.default_rng(image_seed)
    # Separate RNG: IDs contain no shape attributes, targets or target encodings.
    conditions = np.random.default_rng(condition_seed).standard_normal((count, 64)).astype("float32")
    images, masks, records = [], [], []
    for i in range(count):
        bg = rng.integers(15, 55, size=3).tolist()
        image = Image.new("RGB", (48, 48), tuple(bg))
        mask = Image.new("L", (48, 48), 0)
        draw, md = ImageDraw.Draw(image), ImageDraw.Draw(mask)
        shapes = []
        for _ in range(int(rng.integers(2, 5))):
            w, h = [int(v) for v in rng.integers(18, 35, size=2)]
            x, y = int(rng.integers(0, 49-w)), int(rng.integers(0, 49-h))
            bbox = [x, y, x+w-1, y+h-1]
            color = rng.integers(70, 256, size=3).tolist()
            kind = str(rng.choice(["rectangle", "ellipse"]))
            getattr(draw, kind)(bbox, fill=tuple(color))
            getattr(md, kind)(bbox, fill=255)
            shapes.append({"kind": kind, "bbox": bbox, "rgb": color})
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)/255.
        images.append(array)
        masks.append(np.asarray(mask, dtype=np.uint8) > 0)
        image.save(out / "targets" / f"{i:03d}.png")
        mask.save(out / "targets" / f"{i:03d}_mask.png")
        records.append({"id": i, "background_rgb": bg, "shapes": shapes})
    images, masks = np.stack(images), np.stack(masks)
    assert len({hashlib.sha256(x.tobytes()).hexdigest() for x in images}) == count
    assert len(np.unique(conditions, axis=0)) == count
    np.savez(out / "dataset.npz", images=images, masks=masks, conditions=conditions)
    plan = {"count": count, "image_seed": image_seed, "condition_seed": condition_seed,
            "foreground_fraction": float(masks.mean()), "image_size": 48,
            "condition_type": "independent_frozen_random_ID",
            "display_order": list(range(count)), "interpolation_pairs": [[i, i+1] for i in range(0, 16, 2)],
            "alphas": [0., .25, .5, .75, 1.], "unseen_seed": condition_seed+10000, "unseen_count": 16,
            "records": records}
    (out / "manifest.json").write_text(json.dumps(plan, indent=2)+"\n")
    return plan


def load_dataset(path):
    with np.load(Path(path) / "dataset.npz", allow_pickle=False) as d:
        return {k: d[k].copy() for k in d.files}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--image-seed", type=int, default=20261023)
    p.add_argument("--condition-seed", type=int, default=41001)
    a = p.parse_args()
    print(json.dumps(create_dataset(a.out, a.count, a.image_seed, a.condition_seed), indent=2))
