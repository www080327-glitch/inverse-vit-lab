"""C06 source-detail banks. No neural model, no feature inversion claim."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Sequence
import numpy as np

MAGIC = b"C06BANK1"
PREFIX = struct.Struct("<8sI")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _image(image):
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
        raise ValueError("expected nonempty H,W,3 RGB")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError("integer RGB only")
    return a


@dataclass
class PatchBundle:
    patches: np.ndarray
    coordinates_xyxy: np.ndarray
    shape: tuple[int, int, int]
    tile: int


def pack(image, tile=32):
    """Pad with known zero pixels; reversible reshape/permutation, no projection."""
    a = _image(image)
    t = _integer(tile, "tile")
    if t < 1:
        raise ValueError("tile must be positive")
    h, w, c = a.shape
    gh, gw = (h+t-1)//t, (w+t-1)//t
    padded = np.zeros((gh*t, gw*t, c), dtype=a.dtype)
    padded[:h, :w] = a
    patches = padded.reshape(gh, t, gw, t, c).transpose(0, 2, 1, 3, 4)
    patches = np.ascontiguousarray(patches.reshape(gh*gw, t, t, c))
    coords = np.array([(x*t, y*t, min((x+1)*t, w), min((y+1)*t, h))
                       for y in range(gh) for x in range(gw)], dtype="<u4")
    return PatchBundle(patches, coords, a.shape, t)


def unpack(bundle):
    h, w, c = bundle.shape
    t = bundle.tile
    gh, gw = (h+t-1)//t, (w+t-1)//t
    p = bundle.patches
    if p.shape != (gh*gw, t, t, c):
        raise ValueError("inconsistent patches")
    full = p.reshape(gh, gw, t, t, c).transpose(0, 2, 1, 3, 4)
    return full.reshape(gh*t, gw*t, c)[:h, :w].copy()


def save_bank(path, image, tile=32, kind="raw"):
    b = pack(image, tile)
    p = b.patches.astype(b.patches.dtype.newbyteorder("<"), copy=False)
    h, w, _ = b.shape
    header = dict(version=1, shape=list(b.shape), tile=b.tile, dtype=p.dtype.str,
                  grid=[(h+b.tile-1)//b.tile, (w+b.tile-1)//b.tile],
                  layout="row-major-tiles,TY,TX,RGB", coordinates="xyxy-u32le",
                  padding="zero-to-tile; crop-to-shape", kind=kind)
    j = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(PREFIX.pack(MAGIC, len(j)))
        f.write(j)
        f.write(b.coordinates_xyxy.tobytes())
        f.write(p.tobytes())
    return storage_breakdown(path)


def storage_breakdown(path):
    with Path(path).open("rb") as f:
        magic, n = PREFIX.unpack(f.read(PREFIX.size))
        if magic != MAGIC:
            raise ValueError("bad magic")
        m = json.loads(f.read(n))
    h, w, _ = m["shape"]
    t = m["tile"]
    count = m["grid"][0]*m["grid"][1]
    item = np.dtype(m["dtype"]).itemsize
    payload = count*t*t*3*item
    result = dict(total_bytes=Path(path).stat().st_size, header_bytes=PREFIX.size+n,
                  coordinate_bytes=count*16, coefficient_bytes=payload,
                  valid_coefficient_bytes=h*w*3*item,
                  padding_bytes=payload-h*w*3*item, dtype=m["dtype"],
                  tile=t, shape=m["shape"], tiles=count)
    assert result["total_bytes"] == result["header_bytes"]+count*16+payload
    return result


def validate_roi(roi, shape):
    if len(roi) != 4:
        raise ValueError("ROI must be x0,y0,x1,y1")
    x0, y0, x1, y1 = [_integer(v, "ROI coordinate") for v in roi]
    h, w = shape[:2]
    if not (0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h):
        raise ValueError("out-of-image or reversed ROI")
    return x0, y0, x1, y1


class BankReader:
    """Real bounded reads. Counts bytes returned by file.read, not physical IO."""
    def __init__(self, path):
        self.path = Path(path)
        self.f = self.path.open("rb")
        self.bytes_read = 0
        self.read_calls = 0
        self.extents = []
        self.tile_ids = set()
        try:
            magic, n = PREFIX.unpack(self._read(0, PREFIX.size))
            if magic != MAGIC or n > 65536:
                raise ValueError("invalid bank header")
            self.meta = json.loads(self._read(PREFIX.size, n))
            self.h, self.w, c = self.meta["shape"]
            self.t = self.meta["tile"]
            self.gh, self.gw = self.meta["grid"]
            self.dtype = np.dtype(self.meta["dtype"])
            if (self.h < 1 or self.w < 1 or c != 3 or self.t < 1 or
                self.dtype.kind not in "iu" or
                [self.gh, self.gw] != [(self.h+self.t-1)//self.t,
                                      (self.w+self.t-1)//self.t]):
                raise ValueError("invalid bank geometry/dtype")
            self.table_offset = PREFIX.size+n
            self.payload_offset = self.table_offset+self.gh*self.gw*16
            self.tile_bytes = self.t*self.t*3*self.dtype.itemsize
            expected = self.payload_offset+self.gh*self.gw*self.tile_bytes
            if self.path.stat().st_size != expected:
                raise ValueError("truncated bank or trailing bytes")
        except Exception:
            self.f.close()
            raise

    def _read(self, offset, count):
        self.f.seek(offset)
        data = self.f.read(count)
        self.bytes_read += len(data)
        self.read_calls += 1
        self.extents.append([offset, len(data)])
        if len(data) != count:
            raise ValueError("truncated read")
        return data

    def _tile(self, tid):
        gy, gx = divmod(tid, self.gw)
        coords = struct.unpack("<IIII", self._read(self.table_offset+16*tid, 16))
        expected = (gx*self.t, gy*self.t, min((gx+1)*self.t, self.w),
                    min((gy+1)*self.t, self.h))
        if coords != expected:
            raise ValueError("coordinate table mismatch")
        raw = self._read(self.payload_offset+tid*self.tile_bytes, self.tile_bytes)
        self.tile_ids.add(tid)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.t, self.t, 3)

    def read_many(self, rois: Sequence[tuple]):
        boxes = [validate_roi(r, (self.h, self.w, 3)) for r in rois]
        needed = set()
        for x0, y0, x1, y1 in boxes:
            if x0 == x1 or y0 == y1:
                continue
            needed.update(gy*self.gw+gx
                          for gy in range(y0//self.t, (y1-1)//self.t+1)
                          for gx in range(x0//self.t, (x1-1)//self.t+1))
        # Within this call, overlapping ROIs share one disk read per tile.
        cache = {tid: self._tile(tid) for tid in sorted(needed)}
        results = []
        for x0, y0, x1, y1 in boxes:
            out = np.empty((y1-y0, x1-x0, 3), dtype=self.dtype)
            if x0 != x1 and y0 != y1:
                for gy in range(y0//self.t, (y1-1)//self.t+1):
                    for gx in range(x0//self.t, (x1-1)//self.t+1):
                        ax, ay = gx*self.t, gy*self.t
                        lx, ly = max(x0, ax), max(y0, ay)
                        rx, ry = min(x1, ax+self.t), min(y1, ay+self.t)
                        out[ly-y0:ry-y0, lx-x0:rx-x0] = \
                            cache[gy*self.gw+gx][ly-ay:ry-ay, lx-ax:rx-ax]
            results.append(out)
        return results

    def read_roi(self, roi):
        return self.read_many([roi])[0]

    def stats(self):
        return dict(bytes_read=self.bytes_read, read_calls=self.read_calls,
                    unique_tiles=len(self.tile_ids), tile_ids=sorted(self.tile_ids),
                    read_extents=self.extents,
                    byte_metric="bytes returned by Python file.read; cold logical IO",
                    full_file_bytes=self.path.stat().st_size)

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def coarse_and_residual(image, scale=8):
    a = _image(image)
    if a.dtype != np.uint8:
        raise ValueError("coarse+residual format currently requires uint8 RGB")
    s = _integer(scale, "scale")
    if s < 1:
        raise ValueError("scale must be positive")
    h, w, _ = a.shape
    ch, cw = (h+s-1)//s, (w+s-1)//s
    padded = np.zeros((ch*s, cw*s, 3), dtype=np.uint32)
    padded[:h, :w] = a
    total = padded.reshape(ch, s, cw, s, 3).sum(axis=(1, 3), dtype=np.uint64)
    counts_y = np.minimum(s, h-np.arange(ch)*s)
    counts_x = np.minimum(s, w-np.arange(cw)*s)
    coarse = (total//(counts_y[:, None]*counts_x[None, :])[:, :, None]).astype(np.uint8)
    expanded = coarse.repeat(s, axis=0).repeat(s, axis=1)[:h, :w]
    residual = a.astype(np.int16)-expanded.astype(np.int16)
    return coarse, residual


def save_residual_bank(directory, image, tile=32, scale=8):
    t, s = _integer(tile, "tile"), _integer(scale, "scale")
    if t < 1 or s < 1 or t % s:
        raise ValueError("tile must be a positive multiple of scale")
    a = _image(image)
    coarse, residual = coarse_and_residual(a, s)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    cb = save_bank(d/"coarse.bank", coarse, t//s, "integer-floor-mean")
    rb = save_bank(d/"residual.bank", residual, t, "exact-signed-residual")
    manifest = dict(version=1, shape=list(a.shape), scale=s, tile=t, dtype="|u1",
                    coarse="coarse.bank", residual="residual.bank",
                    equation="x = repeat(coarse,scale) + int16_residual; crop H,W",
                    coarse_boundary="mean over valid pixels, floor integer")
    (d/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    mb = (d/"manifest.json").stat().st_size
    return dict(coarse=cb, residual=rb, manifest_bytes=mb,
                coarse_only_bytes=cb["total_bytes"]+mb,
                total_bytes=cb["total_bytes"]+rb["total_bytes"]+mb)


class ResidualReader:
    def __init__(self, directory):
        d = Path(directory)
        raw = (d/"manifest.json").read_bytes()
        self.manifest_bytes = len(raw)
        self.meta = json.loads(raw)
        self.c = BankReader(d/"coarse.bank")
        self.r = None
        try:
            self.r = BankReader(d/"residual.bank")
            self.h, self.w, _ = self.meta["shape"]
            self.s = self.meta["scale"]
            if (self.s < 1 or [self.r.h, self.r.w] != [self.h, self.w] or
                self.r.dtype != np.dtype("<i2") or self.c.dtype != np.dtype("u1") or
                [self.c.h, self.c.w] != [(self.h+self.s-1)//self.s,
                                        (self.w+self.s-1)//self.s]):
                raise ValueError("inconsistent residual bundle")
        except Exception:
            self.c.close()
            if self.r is not None:
                self.r.close()
            raise

    def read_many(self, rois):
        boxes = [validate_roi(r, (self.h, self.w, 3)) for r in rois]
        mapped = [(x0//self.s, y0//self.s, (x1+self.s-1)//self.s,
                   (y1+self.s-1)//self.s) if x0 != x1 and y0 != y1 else (0, 0, 0, 0)
                  for x0, y0, x1, y1 in boxes]
        cs, rs = self.c.read_many(mapped), self.r.read_many(boxes)
        out = []
        for b, m, coarse, residual in zip(boxes, mapped, cs, rs):
            x0, y0, x1, y1 = b
            if x0 == x1 or y0 == y1:
                out.append(np.empty((y1-y0, x1-x0, 3), dtype=np.uint8))
                continue
            expanded = coarse.repeat(self.s, 0).repeat(self.s, 1)
            dx, dy = x0-m[0]*self.s, y0-m[1]*self.s
            fine = expanded[dy:dy+y1-y0, dx:dx+x1-x0].astype(np.int16)+residual
            if fine.min() < 0 or fine.max() > 255:
                raise ValueError("invalid reconstruction range")
            out.append(fine.astype(np.uint8))
        return out

    def read_roi(self, roi):
        return self.read_many([roi])[0]

    def stats(self):
        c, r = self.c.stats(), self.r.stats()
        return dict(bytes_read=self.manifest_bytes+c["bytes_read"]+r["bytes_read"],
                    manifest_bytes=self.manifest_bytes, coarse=c, residual=r,
                    full_file_bytes=self.manifest_bytes+c["full_file_bytes"]+r["full_file_bytes"])

    def close(self):
        self.c.close()
        self.r.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class SemanticInput:
    values: np.ndarray
    producer: str
    synthetic: bool
    source_rgb_sha256: str
    coordinate_space: str = "source_xyxy_half_open"

    def save(self, directory):
        v = np.asarray(self.values)
        if v.ndim != 2 or v.dtype != np.float32 or not np.isfinite(v).all():
            raise ValueError("expected finite float32 token matrix")
        if not self.producer or (self.synthetic and not self.producer.startswith("SYNTHETIC")):
            raise ValueError("synthetic producer must be explicitly marked")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d/"semantic.npy", v, allow_pickle=False)
        meta = dict(producer=self.producer, synthetic=self.synthetic,
                    source_rgb_sha256=self.source_rgb_sha256, dtype="float32",
                    shape=list(v.shape), coordinate_space=self.coordinate_space,
                    file="semantic.npy", qwen_features_used=False if self.synthetic else None)
        (d/"semantic.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return sum((d/n).stat().st_size for n in ("semantic.npy", "semantic.json"))


def bind_source(image, semantic, bank_relative_path, directory):
    """Preserve independent semantic and raw handles; never pretend one inverts the other."""
    digest = hashlib.sha256(_image(image).tobytes()).hexdigest()
    if semantic.source_rgb_sha256 != digest:
        raise ValueError("semantic/source identity mismatch")
    meta = dict(schema="C06-SourceDetailHandle-v1", source_rgb_sha256=digest,
                source_shape=list(image.shape), source_dtype=str(image.dtype),
                raw_bank=bank_relative_path, semantics="semantic.json",
                raw_required=True, coordinates="source pixels, xyxy half-open",
                transform_to_source="identity for this synthetic demo",
                no_source_policy="unavailable; never fabricate source residual")
    p = Path(directory)/"source_handle.json"
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p.stat().st_size


def read_details(handle, directory, rois, expected_rgb_sha256):
    """Recommended integration API; coordinates are source pixels, no semantic inference."""
    if handle["source_rgb_sha256"] != expected_rgb_sha256:
        raise ValueError("source identity mismatch")
    with BankReader(Path(directory)/handle["raw_bank"]) as reader:
        pixels = reader.read_many(rois)
        return pixels, reader.stats()


def read_coarse_only(directory, rois):
    """Standalone lossy baseline; deliberately never opens residual.bank."""
    d = Path(directory)
    raw = (d/"manifest.json").read_bytes()
    m = json.loads(raw)
    s = m["scale"]
    boxes = [validate_roi(b, m["shape"]) for b in rois]
    mapped = [(x0//s, y0//s, (x1+s-1)//s, (y1+s-1)//s)
              if x0 != x1 and y0 != y1 else (0,0,0,0)
              for x0,y0,x1,y1 in boxes]
    with BankReader(d/"coarse.bank") as reader:
        cs = reader.read_many(mapped)
        stats = reader.stats()
    out = []
    for (x0,y0,x1,y1), (cx,cy,_,_), c in zip(boxes,mapped,cs):
        if x0 == x1 or y0 == y1:
            out.append(np.empty((y1-y0,x1-x0,3),np.uint8))
            continue
        e = c.repeat(s,0).repeat(s,1)
        dx,dy = x0-cx*s,y0-cy*s
        out.append(e[dy:dy+y1-y0,dx:dx+x1-x0].copy())
    return out, dict(bytes_read=len(raw)+stats["bytes_read"],
                     manifest_bytes=len(raw), coarse=stats,
                     full_file_bytes=len(raw)+stats["full_file_bytes"])
