"""Validate and package only standalone C11 deliverables (stdlib only)."""
from pathlib import Path
import csv, hashlib, json, os, zipfile, sys

root = Path(__file__).resolve().parent
out = root / "results"
data = json.loads((out / "RESULT.json").read_text())
rows = data["rows"]
assert data["status"] == "completed"
assert data["environment"]["source_sha256"] == hashlib.sha256((root/"runner.mjs").read_bytes()).hexdigest()
assert data["aggregate_wall_seconds"] < 1750
assert len(rows) == 16
assert len({(r["seed"],r["relation"],r["arm"]) for r in rows}) == 16
assert {r["seed"] for r in rows} == {11,29}
assert all(r["parameters"] == 136 and r["teacher_frozen_verified"] for r in rows)
assert all(r["new_updates"] == 800 and r["new_supervised_samples"] == 25600 for r in rows)
for seed in [11,29]:
    rr = [r for r in rows if r["seed"] == seed]
    assert len({r["paired_snapshot_hash"] for r in rr}) == 1
    assert len({r["paired_schedule_hash"] for r in rr}) == 1
    for r in rr:
        assert r["retained_correct"] <= r["teacher_correct"]
        assert abs(r["retention"] - r["retained_correct"]/r["teacher_correct"]) < 1e-12
        assert r["total_updates"] == (800 if r["arm"] == "A" else 1000)
        assert r["old_labeled_samples"] == (3200 if r["arm"] in ["B","D"] else 0)
        assert r["old_distilled_samples"] == (3200 if r["arm"] in ["C","D"] else 0)
        assert r["seconds_training_teacher_forward"] == 0 if r["arm"] in ["A","B"] else r["seconds_training_teacher_forward"] > 0
validation = dict(status="passed", experiment="C11", scope="toy mechanism evidence only",
                 two_paired_seeds=True, rows=16, snapshot_and_schedule_pairing=True,
                 teacher_frozen=True, count_consistency=True,
                 parameter_count=136, gradient_check=data["tests"],
                 pytorch_probe_validation="syntax only; not runtime-tested with PyTorch/Qwen",
                 training_wall_seconds=data["aggregate_wall_seconds"],
                 cpu_affinity=data["environment"]["cpu_affinity"],
                 training_source_sha256=data["environment"]["source_sha256"])
(out / "VALIDATION.json").write_text(json.dumps(validation, indent=2)+"\n")
entries = sorted(p for p in root.rglob("*") if p.is_file()
                 and p.name not in ["C11_DELIVERY.zip", "SHA256SUMS"]
                 and "__pycache__" not in p.parts)
archive_entries = [(p, p.relative_to(root).as_posix()) for p in entries]
workflow = root.parents[1]/".github/workflows/c11-toy.yml"
archive_entries.append((workflow, "c11-toy.yml"))
sums = []
for p, name in archive_entries:
    sums.append(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+name)
(root/"SHA256SUMS").write_text("\n".join(sums)+"\n")
with zipfile.ZipFile(root/"C11_DELIVERY.zip","w",zipfile.ZIP_DEFLATED) as z:
    for p, name in archive_entries+[(root/"SHA256SUMS", "SHA256SUMS")]:
        z.write(p, "C11/"+name)
with zipfile.ZipFile(root/"C11_DELIVERY.zip") as z:
    assert z.testzip() is None
print(json.dumps(validation))
