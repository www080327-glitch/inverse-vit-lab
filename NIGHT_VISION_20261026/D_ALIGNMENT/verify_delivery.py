#!/usr/bin/env python3
"""Verify portable source/evidence integrity without importing torch or a model."""
import ast
import hashlib
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parent
    ready = json.loads((root / "READY.json").read_text())
    assert ready["status"] == "READY_CPU_IMPLEMENTATION_ONLY"
    assert ready["completion"] == {"code": True, "CPU_checks": True, "real_model_training": False}
    checked = 0
    for name, item in ready["files"].items():
        path = root / name
        data = path.read_bytes()
        assert len(data) == item["bytes"], name
        assert hashlib.sha256(data).hexdigest() == item["sha256"], name
        text = data.decode("utf-8")
        assert not re.search(r"/(?:Users|home)/|github_pat_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----", text), name
        if path.suffix == ".py":
            ast.parse(text, filename=name)
        elif path.suffix == ".json":
            json.loads(text)
        checked += 1
    cpu = json.loads((root / "evidence/cpu_checks.json").read_text())
    hf = json.loads((root / "evidence/hf_random_check.json").read_text())
    assert cpu["status"] == "PASS" and cpu["tests_run"] == 15 and not cpu["errors"]
    assert cpu["torch_threads"] == cpu["interop_threads"] == 1 and cpu["elapsed_seconds"] < 300
    assert cpu["fixture_only"] and not cpu["real_model_training_complete"]
    assert hf["status"] == "PASS" and hf["fixture_only"] and hf["threads"] == 1
    assert not hf["full_0_8B_model_loaded"] and not hf["official_pretrained_weights_loaded"]
    print(json.dumps({"status": "PASS", "verified_files": checked, "core_CPU_checks": cpu["tests_run"],
                      "random_HF_fixture": "PASS", "real_model_training_complete": False}))


if __name__ == "__main__":
    main()
