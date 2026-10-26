import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from detail_bypass import (pack, unpack, save_bank, BankReader, coarse_and_residual,
    save_residual_bank, ResidualReader, SemanticInput, bind_source, read_details)

class DetailTests(unittest.TestCase):
    def test_integer_pack_permutation(self):
        rng = np.random.default_rng(6006)
        for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.int32,
                      np.uint32, np.int64, np.uint64):
            for shape in [(1,1,3), (7,11,3), (32,32,3), (33,65,3)]:
                a = rng.integers(0, 127, size=shape).astype(dtype)
                if np.issubdtype(dtype, np.signedinteger):
                    a -= 63
                for tile in (1, 7, 32):
                    b = pack(a[:, ::-1], tile)
                    np.testing.assert_array_equal(unpack(b), a[:, ::-1])
                    self.assertEqual(unpack(b).dtype, a.dtype)
        # Extreme bits survive; no arithmetic in pack/unpack.
        a = np.array([0, 2**64-1, 2**63], dtype=np.uint64).reshape(1,1,3)
        np.testing.assert_array_equal(unpack(pack(a)), a)

    def test_disk_ROI_and_overlap(self):
        rng = np.random.default_rng(61)
        a = rng.integers(0, 256, (67,99,3), dtype=np.uint8)
        boxes = [(31,31,34,34), (32,32,63,65), (97,65,99,67),
                 (0,0,1,1), (20,20,20,22), (0,0,99,67)]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"raw.bank"
            save_bank(p, a)
            with BankReader(p) as reader:
                out = reader.read_many(boxes)
                for b, r in zip(boxes, out):
                    x0,y0,x1,y1 = b
                    np.testing.assert_array_equal(r, a[y0:y1,x0:x1])
                # Each tile is read exactly once in the batch.
                self.assertEqual(reader.read_calls, 2+2*reader.gh*reader.gw)
            with BankReader(p) as reader:
                reader.read_roi((31,31,34,34))
                stats = reader.stats()
                self.assertEqual(stats["unique_tiles"], 4)
                self.assertLess(stats["bytes_read"], p.stat().st_size)
            for bad in [(-1,0,1,1), (1,1,100,68), (3,0,2,2), (.5,0,1,1)]:
                with BankReader(p) as reader:
                    with self.assertRaises(ValueError):
                        reader.read_roi(bad)

    def test_residual_random_boundaries(self):
        rng = np.random.default_rng(62)
        for h,w in [(1,1),(3,5),(31,65),(67,99)]:
            a = rng.integers(0,256,(h,w,3),dtype=np.uint8)
            with tempfile.TemporaryDirectory() as td:
                save_residual_bank(td,a)
                with ResidualReader(td) as reader:
                    full = reader.read_roi((0,0,w,h))
                    np.testing.assert_array_equal(full,a)
                boxes = [(0,0,w,h), (w-1,h-1,w,h), (0,0,0,0)]
                for _ in range(30):
                    xs = sorted(rng.integers(0,w+1,2).tolist())
                    ys = sorted(rng.integers(0,h+1,2).tolist())
                    boxes.append((xs[0],ys[0],xs[1],ys[1]))
                with ResidualReader(td) as reader:
                    for box, crop in zip(boxes,reader.read_many(boxes)):
                        x0,y0,x1,y1 = box
                        np.testing.assert_array_equal(crop,full[y0:y1,x0:x1])
                coarse, r = coarse_and_residual(a)
                self.assertEqual(r.dtype,np.int16)
                if h == w == 1:
                    np.testing.assert_array_equal(coarse,a)

    def test_coordinate_corruption_and_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"raw.bank"
            save_bank(p,np.zeros((35,35,3),np.uint8))
            with BankReader(p) as reader:
                offset = reader.table_offset
            raw = bytearray(p.read_bytes())
            raw[offset] = 1
            p.write_bytes(raw)
            with BankReader(p) as reader:
                with self.assertRaises(ValueError):
                    reader.read_roi((0,0,1,1))
            p.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                BankReader(p)

    def test_semantics_identity_and_separation(self):
        a = np.full((35,37,3),127,np.uint8)
        digest = hashlib.sha256(a.tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            save_bank(Path(td)/"raw.bank",a)
            sem = SemanticInput(np.zeros((4,8),np.float32),
                                "SYNTHETIC-unit-test",True,digest)
            sem.save(td)
            bind_source(a,sem,"raw.bank",td)
            handle = json.loads((Path(td)/"source_handle.json").read_text())
            pixels,stats = read_details(handle,td,[(31,31,35,35)],digest)
            np.testing.assert_array_equal(pixels[0],a[31:35,31:35])
            with self.assertRaises(ValueError):
                read_details(handle,td,[], "wrong source")
            with self.assertRaises(ValueError):
                SemanticInput(np.zeros((1,1),np.float32),"Qwen",True,digest).save(td)
            with self.assertRaises(ValueError):
                bind_source(a,SemanticInput(np.zeros((1,1),np.float32),
                            "SYNTHETIC",True,"wrong"),"raw.bank",td)
            # Semantic values have no influence on source reconstruction.
            SemanticInput(np.ones((4,8),np.float32),"SYNTHETIC-replaced",
                          True,digest).save(td)
            pixels,_ = read_details(handle,td,[(31,31,35,35)],digest)
            np.testing.assert_array_equal(pixels[0],a[31:35,31:35])

    def test_invalid_inputs(self):
        for a in [np.zeros((2,2)), np.zeros((2,2,3),float),
                  np.zeros((0,2,3),np.uint8)]:
            with self.assertRaises(ValueError):
                pack(a)
        with self.assertRaises(ValueError):
            pack(np.zeros((1,1,3),np.uint8),0)
        with self.assertRaises(ValueError):
            coarse_and_residual(np.zeros((1,1,3),np.uint16))
        with self.assertRaises(ValueError):
            save_residual_bank("unused",np.zeros((1,1,3),np.uint8),31,8)

if __name__ == "__main__":
    unittest.main(verbosity=2)
