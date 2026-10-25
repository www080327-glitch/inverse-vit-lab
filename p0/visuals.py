"""Render model outputs, with clipping for PNG display only."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def rgb(value, scale=3):
    array = (np.clip(value.transpose(1, 2, 0), 0, 1)*255).round().astype("uint8")
    return Image.fromarray(array).resize((48*scale, 48*scale), Image.Resampling.NEAREST)


def comparisons(targets, before, after, out):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    for start in range(0, len(targets), 8):
        count = min(8, len(targets)-start)
        canvas = Image.new("RGB", (500, 26+count*155), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 6), "TARGET", fill="black")
        draw.text((195, 6), "BEFORE", fill="black")
        draw.text((350, 6), "AFTER (last step)", fill="black")
        for row, i in enumerate(range(start, start+count)):
            draw.text((2, 35+row*155), str(i), fill="black")
            for col, values in enumerate([targets, before, after]):
                canvas.paste(rgb(values[i]), (30+col*155, 26+row*155))
        canvas.save(out / f"comparison_{start:03d}_{start+count-1:03d}.png")


def diagnostic_grid(values, labels, columns, out):
    rows = (len(values)+columns-1)//columns
    canvas = Image.new("RGB", (columns*155, rows*175), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (value, label) in enumerate(zip(values, labels)):
        x, y = i % columns*155, i//columns*175
        draw.text((x+2, y+3), str(label), fill="black")
        canvas.paste(rgb(value), (x, y+23))
    canvas.save(out)
