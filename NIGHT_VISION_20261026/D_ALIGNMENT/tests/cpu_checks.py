"""Random, single-thread engineering checks. NEVER real-Qwen evidence."""
import inspect
import tempfile
import time
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from alignment.core import Alignment, Condition, FeatureContract, fingerprint, WIDTH
from alignment.backends import TinyBackend, TinyDecoder
from alignment.io import digest, sha, write_json, write_jsonl, save_checkpoint, restore_checkpoint
from alignment.data import Dataset, join_snapshots, replay_sets, score_attributes


def fixture(interface="direct", scope="full", space="raw"):
    torch.manual_seed(2710)
    backbone, decoder = TinyBackend(scope), TinyDecoder()
    mean, std = torch.linspace(-1, 1, WIDTH), torch.linspace(.5, 2, WIDTH)
    return Alignment(backbone, decoder, FeatureContract(mean, std, space), interface, 16)


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_01_shift_and_causality(self):
        engine = fixture()
        raw = torch.randn(1, 64, WIDTH)
        target = engine.contract.normalize(raw)
        suffix = engine.suffix(target[:, :-1])
        self.assertTrue(torch.equal(suffix[:, :1], engine.backbone.output_start(1) + engine.output_coordinates[:1]))
        self.assertTrue(torch.equal(suffix[:, 1:], target[:, :-1] + engine.output_coordinates[1:]))
        condition = Condition("description", "A small red square on the left.")
        first, _ = engine.teacher_forced(condition, raw)
        changed = raw.clone()
        changed[:, 30:] += 100
        second, _ = engine.teacher_forced(condition, changed)
        # Changing z30 and all future targets must not affect predictions z0..z30.
        torch.testing.assert_close(first[:, :31], second[:, :31], rtol=0, atol=0)
        self.assertGreater(float((first[:, 31:] - second[:, 31:]).abs().max().detach()), 1e-5)
        changed = raw.clone()
        changed[:, -1] += 1000
        third, _ = engine.teacher_forced(condition, changed)
        torch.testing.assert_close(first, third, rtol=0, atol=0)

    def test_02_condition_switch_and_no_target_leak(self):
        engine = fixture()
        for mode, image in (("redraw", "synthetic_source"), ("description", None), ("class_only", None)):
            c = Condition(mode, "red", image)
            engine.teacher_forced(c, torch.randn(1, 64, WIDTH))
            self.assertEqual(engine.backbone.seen[-1], c)
        for mode in ("description", "class_only"):
            with self.assertRaises(ValueError):
                Condition(mode, "red", "forbidden_target.png")
        with self.assertRaises(TypeError):
            Condition(**{"mode": "description", "text": "red", "target_features": [1]})
        self.assertNotIn("target", str(inspect.signature(engine.generate)))
        c = Condition("description", "red")
        # Targets have no access path to prepare(condition), even in teacher forcing.
        engine.teacher_forced(c, torch.zeros(1, 64, WIDTH))
        engine.teacher_forced(c, torch.ones(1, 64, WIDTH))
        self.assertEqual(engine.backbone.seen[-2], engine.backbone.seen[-1])

    def test_03_shapes_and_identical_numeric_contract(self):
        a, b = fixture("direct"), fixture("mlp")
        raw = torch.randn(1, 64, WIDTH)
        c = Condition("class_only", "cat")
        pa, ta = a.teacher_forced(c, raw)
        pb, tb = b.teacher_forced(c, raw)
        self.assertEqual(pa.shape, (1, 64, WIDTH))
        self.assertEqual(pb.shape, pa.shape)
        torch.testing.assert_close(ta, tb, rtol=0, atol=0)
        torch.testing.assert_close(a.suffix(ta[:, :-1]), b.suffix(tb[:, :-1]), rtol=0, atol=0)
        torch.testing.assert_close(a.contract.raw(ta), raw)
        torch.testing.assert_close(a.contract.for_decoder(ta), raw)
        model_space = fixture(space="model")
        torch.testing.assert_close(model_space.contract.for_decoder(ta), ta)
        hidden = torch.randn(1, 64, WIDTH)
        self.assertIs(a.head(hidden), hidden)
        self.assertEqual(fingerprint(a.backbone), fingerprint(b.backbone))
        self.assertEqual(fingerprint(a.decoder), fingerprint(b.decoder))
        self.assertEqual(set(n for n, p in a.backbone.named_parameters() if p.requires_grad),
                         set(n for n, p in b.backbone.named_parameters() if p.requires_grad))

    def test_04_frozen_D_gradient_reaches_M_pixel_only(self):
        for interface in ("direct", "mlp"):
            engine = fixture(interface)
            before = {"E": fingerprint(engine.backbone.encoder), "D": fingerprint(engine.decoder)}
            engine.train()
            pred, _ = engine.teacher_forced(Condition("description", "red square"), torch.randn(1, 64, WIDTH))
            pred.retain_grad()
            image = engine.decode(pred)
            image.square().mean().backward()  # NO feature loss, no replay escape route
            self.assertGreater(float(pred.grad.abs().sum()), 0)
            grads = [p.grad for p in engine.backbone.parameters() if p.requires_grad and p.grad is not None]
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in engine.decoder.parameters()))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in engine.backbone.encoder.parameters()))
            opt = torch.optim.AdamW([p for p in engine.parameters() if p.requires_grad], lr=.001)
            opt.step()
            self.assertEqual(before, {"E": fingerprint(engine.backbone.encoder), "D": fingerprint(engine.decoder)})
            self.assertFalse(engine.decoder.training)
            self.assertFalse(engine.backbone.encoder.training)

    def test_05_free_generation_64_uses_own_history(self):
        engine = fixture()
        calls = []
        original = engine.suffix
        def record(history):
            calls.append(history.detach().clone())
            return original(history)
        engine.suffix = record
        generated = engine.generate(Condition("description", "red square"))
        self.assertEqual(generated.shape, (1, 64, WIDTH))
        self.assertEqual([t.shape[1] for t in calls], list(range(64)))
        for i, history in enumerate(calls):
            torch.testing.assert_close(history, generated[:, :i], rtol=0, atol=0)
        self.assertEqual(len(engine.backbone.seen), 1)

    def test_06_scopes(self):
        for interface in ("direct", "mlp"):
            engine = fixture(interface, "last4")
            names = {n for n, p in engine.backbone.named_parameters() if p.requires_grad}
            self.assertTrue(names)
            self.assertTrue(all(n.startswith(tuple(f"layers.{i}." for i in range(2, 6))) for n in names))
            self.assertFalse(engine.backbone.text_head.weight.requires_grad)
            self.assertEqual(engine.backbone.text_head.out_features, 128)
            engine.losses(Condition("description", "red"), torch.randn(1, 64, WIDTH))[0].backward()
            self.assertTrue(any(p.grad is not None for p in engine.backbone.layers[-1].parameters()))
            self.assertTrue(all(p.grad is None for p in engine.backbone.layers[0].parameters()))

    def test_07_save_restore_optimizer_next_step(self):
        for interface in ("direct", "mlp"):
            a = fixture(interface)
            opt = torch.optim.AdamW([p for p in a.parameters() if p.requires_grad], lr=.001)
            c, raw = Condition("class_only", "cat"), torch.randn(1, 64, WIDTH)
            a.losses(c, raw)[0].backward()
            opt.step()
            checkpoint = self.root / interface
            save_checkpoint(a, opt, checkpoint, 1, "fixture_only")
            rng = torch.get_rng_state().clone()
            b = fixture(interface)
            opt_b = torch.optim.AdamW([p for p in b.parameters() if p.requires_grad], lr=.001)
            meta = restore_checkpoint(b, opt_b, checkpoint, "fixture_only")
            self.assertEqual(meta["step"], 1)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(fingerprint(a), fingerprint(b))
            for engine, optimizer in ((a, opt), (b, opt_b)):
                optimizer.zero_grad(set_to_none=True)
                engine.losses(c, raw)[0].backward()
                optimizer.step()
            self.assertEqual(fingerprint(a), fingerprint(b))
            with self.assertRaises(ValueError):
                restore_checkpoint(b, opt_b, checkpoint, "wrong_manifest")

    def test_08_independent_decode_without_M_E_targets(self):
        from alignment.runner import independent_decode
        from alignment.backends import load_decoder
        source = Path(inspect.getfile(TinyDecoder))
        decoder = TinyDecoder()
        weights = self.root / "D.safetensors"
        save_file(decoder.state_dict(), str(weights))
        normal = {"kind": "raw", "decoder_space": "raw"}
        rep = {"fixture": True}
        c = {"normalization": normal, "representation": rep,
             "decoder": {"weights": {"path": str(weights), "sha256": sha(weights)},
                         "source": {"path": str(source), "sha256": sha(source)},
                         "factory": "alignment.backends:TinyDecoder", "kwargs": {}, "call": "features",
                         "representation_sha256": digest(rep)}}
        z = torch.randn(1, 64, WIDTH)
        features, output = self.root / "features.safetensors", self.root / "decoded.safetensors"
        save_file({"features_model_space": z, "features_raw": z.clone()}, str(features))
        independent_decode(c, self.root, features, output)
        torch.testing.assert_close(load_file(str(output))["prediction_rgb_unclipped"], decoder(z))
        bad = self.root / "leak.safetensors"
        save_file({"features_model_space": z, "features_raw": z.clone(), "target": z.clone()}, str(bad))
        with self.assertRaises(ValueError):
            independent_decode(c, self.root, bad, self.root / "bad.safetensors")

    def _snapshots(self):
        image_rows, descriptions = [], []
        rep = {"fixture": True}
        source_model = {"repository": "Qwen/Qwen3.5-0.8B", "revision": "fixture_not_real_inference"}
        for i, split in enumerate(("train", "train", "val")):
            image = self.root / f"image{i}.fixture"
            image.write_text(f"RANDOM_FIXTURE_{i}")
            features = self.root / f"feature{i}.safetensors"
            save_file({"features": torch.randn(64, WIDTH)}, str(features))
            def spec(path):
                return {"path": str(path), "sha256": sha(path)}
            row = {"image_id": f"i{i}", "split": split, "image": spec(image),
                   "source_features": spec(features), "class_word": "square"}
            image_rows.append(row)
            descriptions.append({"image_id": row["image_id"], "image_sha256": row["image"]["sha256"],
                "split": split, "raw_description": "A red square.", "uncertainty": ["layout unspecified"],
                "structure": {"subject": {"value": "square", "explicit_in_description": True}},
                "source_model": source_model})
        split_sha = digest([{"image_id": r["image_id"], "image_sha256": r["image"]["sha256"], "split": r["split"]} for r in image_rows])
        paths = []
        for kind, rows in (("image_features", image_rows), ("original_qwen_descriptions", descriptions)):
            path = self.root / f"{kind}.jsonl"
            write_jsonl(path, rows)
            meta = {"schema": "alignment.snapshot.v1", "status": "complete", "kind": kind,
                    "records": {"path": path.name, "sha256": sha(path)}, "counts": {"train": 2, "val": 1},
                    "split_sha256": split_sha, "representation": rep, "source_model": source_model}
            commit = self.root / f"{kind}.json"
            write_json(commit, meta)
            paths.append(commit)
        return paths, image_rows, descriptions, rep

    def test_09_immutable_join_and_inference_separation(self):
        paths, images, descriptions, rep = self._snapshots()
        joined = self.root / "joined"
        join_snapshots(*paths, joined)
        commit = joined / "snapshot.json"
        data = Dataset({"path": str(commit), "sha256": sha(commit)}, self.root, rep, {"train": 2, "val": 1})
        row = data.rows[-1]
        self.assertEqual(row["description"], descriptions[-1])
        # Remove reference files after opening snapshot: text-only inference
        # condition creation still succeeds, redraw and scoring fail closed.
        Path(row["image"]["path"]).unlink()
        Path(row["source_features"]["path"]).unlink()
        self.assertIsNone(data.condition(row, "description").image)
        self.assertIsNone(data.condition(row, "class_only").image)
        with self.assertRaises(FileNotFoundError):
            data.condition(row, "redraw")
        with self.assertRaises(FileNotFoundError):
            data.target(row, "cpu")
        Path(paths[1]).write_text(Path(paths[1]).read_text().replace('"complete"', '"writing"'))
        with self.assertRaises(ValueError):
            join_snapshots(*paths, self.root / "bad")

    def test_10_join_rejects_hash_and_split_mismatch(self):
        paths, images, descriptions, rep = self._snapshots()
        descriptions[-1]["image_sha256"] = "0" * 64
        records = self.root / "original_qwen_descriptions.jsonl"
        write_jsonl(records, descriptions)
        from alignment.io import read_json
        meta = read_json(paths[1])
        meta["records"]["sha256"] = sha(records)
        write_json(paths[1], meta)
        with self.assertRaises(ValueError):
            join_snapshots(*paths, self.root / "bad_hash")

    def test_11_instruction_replay_disjoint_and_keeps_head(self):
        train, evaluation = self.root / "train.jsonl", self.root / "eval.jsonl"
        write_jsonl(train, [{"id": "t", "split": "train", "prompt": "Train question", "response": "answer"}])
        write_jsonl(evaluation, [{"id": "e", "split": "eval", "prompt": "Held out question", "rubric": "Follow the instruction"}])
        spec = {k: {"path": str(v), "sha256": sha(v)} for k, v in (("train", train), ("eval", evaluation))}
        a, b = replay_sets(spec, self.root)
        engine = fixture()
        engine.backbone.replay_loss(a[0]).backward()
        self.assertIsNotNone(engine.backbone.text_head.weight.grad)
        write_jsonl(evaluation, [{"id": "e", "split": "eval", "prompt": " train   QUESTION ", "rubric": "rubric"}])
        spec["eval"]["sha256"] = sha(evaluation)
        with self.assertRaises(ValueError):
            replay_sets(spec, self.root)

    def test_12_attribute_metrics_explicit_only(self):
        queue, judged = self.root / "queue.jsonl", self.root / "judged.jsonl"
        write_jsonl(queue, [{"key": "description_0", "mode": "description", "prediction_sha256": "a"*64,
                            "explicit_attributes": {"subject": "square", "color": "red"}}])
        write_jsonl(judged, [{"key": "description_0", "prediction_sha256": "a"*64, "judge": "SYNTHETIC_FIXTURE",
                             "scores": {"subject": 1, "color": 0}}])
        result = score_attributes(queue, judged, self.root / "metrics.json")
        self.assertEqual(result["metrics"]["description/subject"]["satisfaction_rate"], 1)
        self.assertNotIn("description/layout", result["metrics"])

    def test_13_real_execution_guard(self):
        from alignment.runner import require_A
        for executor, device in (("local", "cuda"), ("A", "cpu"), ("A", "mps")):
            with self.assertRaises(ValueError):
                require_A(executor, device)

    def test_14_manifest_contract_and_scope_lock(self):
        from alignment.io import read_json
        from alignment.runner import manifest
        c = read_json(Path(__file__).parents[1] / "configs" / "paired_short.example.json")
        c["encoder"]["representation_sha256"] = digest(c["representation"])
        c["decoder"]["representation_sha256"] = digest(c["representation"])
        c["decoder"]["normalization_sha256"] = digest(c["normalization"])
        c["resource_lock"] = {"status": "locked", "executor": "A", "scope": "last4"}
        c["training"]["scope"] = "last4"
        path = self.root / "manifest.json"
        write_json(path, c)
        parsed, _ = manifest(path)
        self.assertEqual(parsed["training"]["scope"], "last4")
        c["normalization"]["decoder_space"] = "model"
        write_json(path, c)
        with self.assertRaises(ValueError):
            manifest(path)
        c["normalization"]["decoder_space"] = "raw"
        c["training"]["scope"] = "full"
        write_json(path, c)
        with self.assertRaises(ValueError):
            manifest(path)

    def test_15_paired_runner_fixture_end_to_end(self):
        from unittest.mock import patch
        from alignment.runner import run_pair
        from alignment.io import read_json
        c = read_json(Path(__file__).parents[1] / "configs" / "paired_short.example.json")
        c["training"].update(steps=2, checkpoint_every=1)
        c["evaluation"]["val_samples"] = 1
        c["limits"]["wall_seconds_per_interface"] = 30
        c["replay"]["every"] = 1
        class ToyData:
            splits = {"train": [{"image_id": "t0"}, {"image_id": "t1"}],
                      "val": [{"image_id": "v0", "class_word": "square", "description": {
                          "structure": {"subject": {"value": "square", "explicit_in_description": True}}}}]}
            raw = torch.zeros(1, 64, WIDTH)
            rgb = torch.zeros(1, 3, 8, 8)
            def condition(self, row, mode):
                return Condition(mode, "square", "synthetic" if mode == "redraw" else None)
            def target(self, row, device, include_rgb=False):
                return self.raw, self.rgb if include_rgb else None
        def toy_build(config, root, interface, device):
            e = fixture(interface)
            e.backbone.runtime = {"fixture_only": True}
            e.backbone.instruction_response = lambda row, max_tokens: "RANDOM_FIXTURE_REPLY"
            return e, fingerprint(e.backbone)
        replay = [{"id": "rt", "prompt": "training", "response": "toy"}]
        evaluation = [{"id": "re", "prompt": "held out", "rubric": "fixture"}]
        with patch("alignment.runner.validate_inputs", return_value=(ToyData(), replay, evaluation)), \
             patch("alignment.runner.build", side_effect=toy_build), \
             patch("torch.cuda.manual_seed_all"), patch("torch.cuda.empty_cache"), \
             patch("torch.cuda.max_memory_allocated", return_value=0), \
             patch("torch.cuda.max_memory_reserved", return_value=0):
            run_pair(c, self.root, self.root / "paired", "cpu")
        report = read_json(self.root / "paired" / "pair_result.json")
        self.assertTrue(report["matched"])
        for interface in ("direct", "mlp"):
            output = self.root / "paired" / interface
            self.assertEqual(read_json(output / "status.json")["optimizer_steps"], 2)
            self.assertEqual(len(read_json(output / "evaluation" / "metrics.json")["rows"]), 3)
            self.assertTrue((output / "step_0002" / "checkpoint.json").exists())
            self.assertTrue((output / "instruction_before.jsonl").exists())
            self.assertTrue((output / "instruction_after.jsonl").exists())


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started = time.monotonic()
        super().startTest(test)

    def stopTest(self, test):
        failed = any(t == test for t, _ in self.failures + self.errors)
        self.records.append({"name": test._testMethodName, "passed": not failed,
                             "seconds": time.monotonic() - self.started})
        super().stopTest(test)


def run(output):
    import sys
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    self_result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=Result).run(suite)
    result = {"status": "PASS" if self_result.wasSuccessful() else "FAIL", "fixture_only": True,
              "real_Qwen_loaded": False, "DCU_connected": False, "MPS_used": False,
              "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "torch_version": torch.__version__, "elapsed_seconds": time.monotonic() - start,
              "tests_run": self_result.testsRun, "checks": self_result.records,
              "errors": [str(e) for _, e in self_result.errors + self_result.failures],
              "real_model_training_complete": False}
    write_json(output, result)
    if not self_result.wasSuccessful():
        raise RuntimeError("CPU fixture checks failed; READY must not be published")
    return result
