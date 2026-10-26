"""Immutable B snapshot join and strict separation of conditioning from targets."""
from pathlib import Path
from .core import Condition, WIDTH, TOKENS
from .io import digest, jsonl, locked_file, read_json, sha, write_json, write_jsonl


def snapshot(path):
    path = Path(path).resolve()
    commit = read_json(path)
    if commit["status"] != "complete" or commit["schema"] != "alignment.snapshot.v1":
        raise ValueError("Only complete immutable snapshots may be consumed")
    records = locked_file(commit["records"], path.parent)
    return commit, jsonl(records), records.parent


def join_snapshots(images, descriptions, output):
    """No scoring, filtering, relabeling or new split. Exact ID + original-image hash join."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Refuse to replace an existing joined snapshot")
    a, rows, ar = snapshot(images)
    b, desc, br = snapshot(descriptions)
    if a["kind"] != "image_features" or b["kind"] != "original_qwen_descriptions":
        raise ValueError("Wrong snapshot kinds")
    if b["source_model"]["repository"] != "Qwen/Qwen3.5-0.8B" or not b["source_model"].get("revision"):
        raise ValueError("B descriptions must identify the original Qwen source")
    if a["split_sha256"] != b["split_sha256"]:
        raise ValueError("B and feature snapshots must share the authoritative split lock")
    if len({r["image_id"] for r in rows}) != len(rows) or len({r["image_id"] for r in desc}) != len(desc):
        raise ValueError("Duplicate image IDs")
    by_id = {r["image_id"]: r for r in desc}
    if set(by_id) != {r["image_id"] for r in rows}:
        raise ValueError("Snapshot join must be complete; no description-based selection")
    joined = []
    for row in rows:
        d = by_id[row["image_id"]]
        if row["image"]["sha256"] != d["image_sha256"] or row["split"] != d["split"]:
            raise ValueError("Image hash or split mismatch")
        if d["source_model"] != b["source_model"]:
            raise ValueError("Description source changed within snapshot")
        if not isinstance(d["raw_description"], str) or not d["raw_description"].strip():
            raise ValueError("Keep uncertain descriptions, but empty descriptions need a new complete upstream snapshot")
        for field in ("structure", "uncertainty", "source_model"):
            if field not in d:
                raise ValueError(f"Missing original description field {field}")
        out = dict(row)
        for field in ("image", "source_features", "rgb_target"):
            if field in out:
                locked = locked_file(out[field], ar)
                out[field] = {**out[field], "path": str(locked)}
        out["description"] = d  # preserve original raw text, structure, uncertainty and provenance
        joined.append(out)
    counts = {s: sum(r["split"] == s for r in joined) for s in ("train", "val")}
    if counts != a["counts"] or counts != b["counts"] or sum(counts.values()) != len(joined):
        raise ValueError("Split counts mismatch")
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    write_jsonl(records, joined)
    commit = {"schema": "alignment.snapshot.v1", "status": "complete", "kind": "joined",
              "records": {"path": records.name, "sha256": sha(records)}, "counts": counts,
              "split_sha256": a["split_sha256"], "representation": a["representation"],
              "parents": {"image_snapshot_sha256": sha(images), "B_snapshot_sha256": sha(descriptions)},
              "description_status": "original_qwen_candidate_bridge_NOT_stage2_self_description"}
    write_json(output / "snapshot.json", commit)  # commit LAST
    return commit


class Dataset:
    def __init__(self, spec, root, representation, counts):
        path = locked_file(spec, root)
        self.commit, self.rows, self.root = snapshot(path)
        if self.commit["kind"] != "joined" or self.commit["counts"] != counts:
            raise ValueError("Require the predeclared complete joined train/val snapshot")
        if self.commit["representation"] != representation:
            raise ValueError("Feature/encoder/decoder representation mismatch")
        if len({r["image_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate IDs")
        image_hashes = [r["image"]["sha256"] for r in self.rows]
        if len(set(image_hashes)) != len(image_hashes):
            raise ValueError("Duplicate image content across the authoritative dataset")
        self.splits = {s: [r for r in self.rows if r["split"] == s] for s in counts}
        if {s: len(v) for s, v in self.splits.items()} != counts:
            raise ValueError("Actual split counts mismatch")
        split_lock = [{"image_id": r["image_id"], "image_sha256": r["image"]["sha256"],
                       "split": r["split"]} for r in self.rows]
        if digest(split_lock) != self.commit["split_sha256"]:
            raise ValueError("Authoritative split lock mismatch")

    def condition(self, row, mode):
        if mode == "redraw":
            path = locked_file(row["image"], self.root)
            return Condition(mode, "Redraw the source image as a visual sequence.", str(path))
        if mode == "description":
            return Condition(mode, row["description"]["raw_description"])
        if mode == "class_only":
            return Condition(mode, row["class_word"])
        raise ValueError("Unknown condition")

    def target(self, row, device, include_rgb=False):
        import torch
        from safetensors.torch import load_file
        path = locked_file(row["source_features"], self.root)
        payload = load_file(str(path))
        raw = payload[row["source_features"].get("key", "features")]
        if raw.shape == (TOKENS, WIDTH):
            raw = raw.unsqueeze(0)
        if raw.shape != (1, TOKENS, WIDTH) or not torch.isfinite(raw).all():
            raise ValueError("Expected finite raw post-merger feature [64,1024]")
        rgb = None
        if include_rgb:
            path = locked_file(row["rgb_target"], self.root)
            rgb = load_file(str(path))[row["rgb_target"].get("key", "rgb")]
            if rgb.shape == (3, 256, 256):
                rgb = rgb.unsqueeze(0)
            if rgb.shape != (1, 3, 256, 256) or not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
                raise ValueError("RGB scoring target must be exact processed RGB [1,3,256,256] in [0,1]")
            rgb = rgb.to(device)
        return raw.to(device).float(), rgb


def replay_sets(spec, root):
    train = jsonl(locked_file(spec["train"], root))
    evaluation = jsonl(locked_file(spec["eval"], root))
    if not train or not evaluation:
        raise ValueError("Independent instruction replay train AND evaluation sets required")
    seen = set()
    for expected, rows in (("train", train), ("eval", evaluation)):
        for row in rows:
            if row["split"] != expected or not row["prompt"].strip():
                raise ValueError("Instruction split/prompt mismatch")
            key = " ".join(row["prompt"].split()).casefold()
            if row["id"] in seen or key in seen:
                raise ValueError("Replay/evaluation ID or prompt overlap")
            seen.update([row["id"], key])
            if expected == "train" and not row["response"].strip():
                raise ValueError("Replay response required")
            if expected == "eval" and not row.get("rubric"):
                raise ValueError("Instruction evaluation requires a rubric, not CE alone")
    return train, evaluation


def attribute_items(row):
    # Only attributes explicitly supported by the original text; never infer a
    # category label or feature from the held-out reference image.
    fields = row["description"]["structure"]
    result = {}
    for name in ("subject", "color", "layout"):
        item = fields.get(name)
        if isinstance(item, dict) and item.get("explicit_in_description") is True and item.get("value"):
            result[name] = item["value"]
    return result


def score_attributes(queue_path, judgments_path, output):
    queue, judgments = jsonl(queue_path), jsonl(judgments_path)
    expected = {r["key"]: r for r in queue}
    if len(expected) != len(queue) or {r["key"] for r in judgments} != set(expected) or len(judgments) != len(queue):
        raise ValueError("All queued images must be judged exactly once")
    totals = {}
    for row in judgments:
        ref = expected[row["key"]]
        if row["prediction_sha256"] != ref["prediction_sha256"] or not row.get("judge"):
            raise ValueError("Judgments must identify the actual generated image and judge")
        if set(row["scores"]) != set(ref["explicit_attributes"]):
            raise ValueError("Only, and all, explicitly described attributes may be scored")
        for attr, score in row["scores"].items():
            if score not in (0, 1) or isinstance(score, bool):
                raise ValueError("Attribute score is 0 (not satisfied) or 1 (satisfied)")
            key = ref["mode"] + "/" + attr
            totals.setdefault(key, []).append(score)
    metrics = {k: {"n": len(v), "satisfaction_rate": sum(v) / len(v)} for k, v in totals.items()}
    result = {"queue_sha256": sha(queue_path), "judgments_sha256": sha(judgments_path),
              "metrics": metrics, "scope": "Only explicit subject/color/layout facts; no exact-pixel requirement"}
    write_json(output, result)
    return result
