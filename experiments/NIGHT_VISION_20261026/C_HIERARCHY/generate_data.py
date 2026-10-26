"""Offline scenes and labels. Model inference never imports this module."""
import itertools
import hashlib
from PIL import Image, ImageDraw
from runtime import ROOT, np, write_json

TYPES = ['eye', 'arm', 'marker', 'connection']
ANSWERS = [['closed', 'open'], ['down', 'up'], ['left', 'right'], ['aligned', 'offset']]

def glyph(kind, state, blind=False, shift=0):
    a = np.zeros((8, 8), dtype=np.uint8)
    if blind:
        # Every type AND answer has exactly the same 2x2-cell means.
        patterns = [[0, 3], [1, 2], [0, 1], [0, 2], [2, 3], [1, 3]]
        for p in patterns[kind]:
            a[2+p//2, 2+p%2] = 1
        # Answer occupies one pixel of another 2x2 cell, without changing its mean.
        a[4+(state if kind < 2 else 0), 4+(state if kind >= 2 else 0)] = 1
        return a
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    if kind == 0:
        if state: d.ellipse((1, 2, 6, 5), outline=1)
        else: d.line((1, 3, 6, 3), fill=1)
    elif kind == 1:
        d.line((3, 2, 3, 6), fill=1)
        y = 1 if state else 5
        d.line((1, y, 3, 3, 5, y), fill=1)
    elif kind == 2:
        d.rectangle((1, 1, 6, 6), outline=1)
        x = 4 if state else 2
        d.rectangle((x, 3, x+1, 4), fill=1)
    elif kind == 3:
        d.line((1, 2, 3, 2, 3, 5, 6, 5), fill=1)
        d.line((4 if state else 3, 1, 4 if state else 3, 6), fill=1)
    else:
        # Brighter and bigger distractors; never a unique maximum for the query.
        d.rectangle((1, 1, 6, 6), fill=1)
        if kind == 5: d.rectangle((2, 2, 5, 5), fill=0)
    a = np.asarray(im)
    return np.roll(a, shift, axis=0)

def make_scene(rng, split, index, blind=False):
    parity = 0 if split == 'train' else 1
    combos = [x for x in itertools.product(range(2), repeat=4) if sum(x)%2 == parity]
    states = list(combos[index % len(combos)])
    # Normal train and test objects use disjoint checkerboard tile locations.
    candidates = list(range(16)) if blind else [r for r in range(16) if (r//4+r%4)%2 == parity]
    slots = rng.choice(candidates, 4, replace=False).tolist()
    kinds = [int(rng.integers(4, 6)) for _ in range(16)]
    cellstates = [int(rng.integers(2)) for _ in range(16)]
    for kind, slot in enumerate(slots): kinds[slot], cellstates[slot] = kind, states[kind]
    # All four target types have identical luminance/color. Shapes are necessary.
    bg = 24 if blind else int(rng.integers(18, 31))
    fg = 196 if blind else int(rng.integers(175, 216))
    image = np.full((32, 32, 3), bg, dtype=np.uint8)
    masks, boxes = [], []
    for r in range(16):
        shift = 0 if blind else int(rng.choice([-1, 0, 1]))
        mask = glyph(kinds[r], cellstates[r], blind, shift)
        color = fg if kinds[r] < 4 or blind else min(250, fg+30)
        y, x = (r//4)*8, (r%4)*8
        image[y:y+8, x:x+8] = np.where(mask[..., None] != 0, color, bg)
        if kinds[r] < 4:
            yy, xx = np.where(mask)
            boxes.append([int(x+xx.min()), int(y+yy.min()), int(x+xx.max()+1), int(y+yy.max()+1)])
            masks.append((r, int(mask.sum())))
    info = {r: count for r, count in masks}
    return image, dict(regions=slots, answers=states, region_kinds=[min(k,4) for k in kinds],
                       target_pixels=[info[r] for r in slots], state_combo=states,
                       region_layout=slots, summary_blind=blind, boxes=boxes)

def generate():
    root = ROOT/'data'
    if (root/'manifest.json').exists():
        print('Existing fixed dataset kept.'); return
    rng = np.random.default_rng(20261026)
    hashes = {}
    for split, n, blind in [('train', 320, False), ('train_blind', 80, True),
                            ('test', 96, False), ('test_blind', 96, True)]:
        folder = root/'images'/split; folder.mkdir(parents=True, exist_ok=True)
        inputs, annotations = [], []
        for i in range(n):
            img, ann = make_scene(rng, 'train' if split.startswith('train') else 'test', i, blind)
            name = f'{split}_{i:04d}'
            path = folder/f'{name}.png'; Image.fromarray(img).save(path)
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            inputs.append(dict(id=name, image=str(path.relative_to(ROOT)), questions=list(range(4))))
            annotations.append(dict(id=name, **ann))
        for role, rows in [('inputs', inputs), ('annotations', annotations)]:
            path = root/role/f'{split}.jsonl'; path.parent.mkdir(exist_ok=True)
            import json
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write_json(root/'manifest.json', dict(seed=20261026, image_sha256=hashes,
               train_scenes=400, test_scenes=192, train_state_parity=0, test_state_parity=1,
               inputs_contain_labels=False, pixels_per_image=1024,
               questions=TYPES, answers=ANSWERS,
               normal_layout_split='train even checkerboard; test odd checkerboard',
               blind_limitation='2x2 means identical across all 16 tiles, all types and states'))
    print('Saved 592 original PNG images; test answers are in separate annotations files.')

if __name__ == '__main__': generate()
