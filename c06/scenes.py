"""Fixed original procedural data: no fonts, external images, or models."""
import hashlib
import json
from pathlib import Path
import struct
import zlib
import numpy as np

GLYPHS = {
    "E": ["111","100","110","100","111"],
    "1": ["010","110","010","010","111"],
    "7": ["111","001","010","010","010"],
}


def png(path, rgb):
    a = np.asarray(rgb, dtype=np.uint8)
    h, w, _ = a.shape
    def chunk(kind, data):
        return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data)&0xffffffff)
    rows = b"".join(b"\0"+a[y].tobytes() for y in range(h))
    body = b"\x89PNG\r\n\x1a\n"
    body += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    body += chunk(b"IDAT", zlib.compress(rows, 6))+chunk(b"IEND", b"")
    Path(path).write_bytes(body)


def draw_scene(h, w, variant):
    yy, xx = np.indices((h, w))
    a = np.empty((h, w, 3), dtype=np.uint8)
    a[..., 0] = 28+(xx//64)%24
    a[..., 1] = 35+(yy//64)%24
    a[..., 2] = 49+(xx//128+yy//128)%24
    a[24:h-20, 24:w-24] = (225, 228, 232)
    a[24:52, 24:w-24] = (59, 68, 85)
    a[70:h-40, 40:w//3] = (245, 246, 248)
    a[70:h-40, w//3+12:w-40] = (251, 251, 251)
    for y in range(90, min(h-80, 700), 24):
        a[y:y+3, 52:min(250, w//3-12)] = (185, 191, 204)
    # Tiny 3x5 hard-coded glyphs. Deliberately no anti-aliasing.
    gx, gy = 319, 191
    text = "E1" if variant == 0 else "E7"
    for i, ch in enumerate(text):
        for dy, row in enumerate(GLYPHS[ch]):
            for dx, p in enumerate(row):
                if p == "1":
                    a[gy+dy, gx+i*4+dx] = (10, 10, 10)
    # One-pixel diagonal and vertical fine lines.
    for k in range(37):
        a[255+k, 447+k] = (12, 32, 60)
    a[255:294, 479] = (180, 20, 20)
    # Eyes: equal black-pixel count per 8x8 coarse cell.
    ex, ey = 512, 288
    a[ey:ey+8, ex:ex+16] = (240, 210, 185)
    pts = [(1, 2), (2, 1), (3, 2), (2, 3)] if variant == 0 \
        else [(0, 2), (1, 2), (2, 2), (3, 2)]
    for eye in (0, 8):
        for dx, dy in pts:
            a[ey+dy, ex+eye+dx] = (0, 0, 0)
    # Tiny accessory: 2x2 square vs 1x4 clip, same per-cell sum.
    ax, ay = 640, 384
    a[ay:ay+8, ax:ax+8] = (200, 215, 245)
    points = [(1, 1), (2, 1), (1, 2), (2, 2)] if variant == 0 \
        else [(0, 1), (1, 1), (2, 1), (3, 1)]
    for dx, dy in points:
        a[ay+dy, ax+dx] = (30, 10, 120)
    # Border target sits at the final odd pixel, testing partial edge tiles.
    a[h-3:h, w-3:w] = (255, 0, 210)
    a[h-1, w-1] = (0, 255, 2)
    targets = [
        dict(name="tiny_glyph", roi=[gx-1, gy-1, gx+8, gy+6]),
        dict(name="one_pixel_lines", roi=[446, 254, 485, 295]),
        dict(name="eyes_open" if variant == 0 else "eyes_closed",
             roi=[ex, ey, ex+16, ey+8]),
        dict(name="accessory_square" if variant == 0 else "accessory_clip",
             roi=[ax, ay, ax+8, ay+8]),
        dict(name="odd_edge", roi=[w-5, h-5, w, h]),
    ]
    for target in targets:
        x0, y0, x1, y1 = target["roi"]
        target["image_fraction"] = (x1-x0)*(y1-y0)/(h*w)
        assert target["image_fraction"] < .01
    return a, targets


def generate(directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    records = []
    for h, w in [(1025, 1537), (769, 1023)]:
        for variant in (0, 1):
            a, targets = draw_scene(h, w, variant)
            name = f"desktop_{w}x{h}_v{variant}"
            png(d/f"{name}.png", a)
            records.append(dict(name=name, height=h, width=w, variant=variant,
                                targets=targets,
                                rgb_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    (d/"SCENES.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records
