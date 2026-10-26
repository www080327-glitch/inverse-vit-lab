"""Weight-independent layout derived from local Qwen2VL patchify + Qwen3.5 merger.

Official patch index = (((gy * (W/2) + gx) * 2 + sy) * 2 + sx).
The merger consumes four contiguous entries. Decoder patch vectors use HWC RGB,
whereas official processor input vectors use C,temporal,patch_y,patch_x.
Only still images (grid_t=1) are supported; two identical temporal slices are
required when recovering the preprocessed target from processor patch vectors.
"""
import torch


def image_grid(grid_thw, merge_size=2):
    if len(grid_thw) != 3 or any(type(v) is not int for v in grid_thw):
        raise ValueError("grid_thw must be a CPU tuple/list of three Python integers")
    t, h, w = grid_thw
    if t != 1 or h <= 0 or w <= 0 or h % merge_size or w % merge_size:
        raise ValueError("Only positive still-image grids divisible by merge_size are supported")
    return h, w


def grouped_coordinates(grid_thw, merge_size=2):
    h, w = image_grid(grid_thw, merge_size)
    return [(gy*merge_size+sy, gx*merge_size+sx)
            for gy in range(h//merge_size) for gx in range(w//merge_size)
            for sy in range(merge_size) for sx in range(merge_size)]


def grouped_to_raster(x, grid_thw, merge_size=2):
    """[B, merged_N, four, D] -> [B, H*W, D], with explicit global recovery."""
    h, w = image_grid(grid_thw, merge_size)
    b, n, s, d = x.shape
    m = merge_size
    if n != (h//m)*(w//m) or s != m*m:
        raise ValueError("Unmerged tensor does not match grid")
    return x.reshape(b, h//m, w//m, m, m, d).permute(0, 1, 3, 2, 4, 5).reshape(b, h*w, d)


def raster_to_grouped(x, grid_thw, merge_size=2):
    h, w = image_grid(grid_thw, merge_size)
    b, n, d = x.shape
    m = merge_size
    if n != h*w:
        raise ValueError("Raster length does not match grid")
    return x.reshape(b, h//m, m, w//m, m, d).permute(0, 1, 3, 2, 4, 5).reshape(b, h*w//(m*m), m*m, d)


def unpatchify(patches, grid_thw, patch_size=16):
    h, w = image_grid(grid_thw)
    b = patches.shape[0]
    p = patch_size
    if patches.shape != (b, h*w, p*p*3):
        raise ValueError("Expected raster patches with HWC RGB vectors")
    return patches.reshape(b, h, w, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, h*p, w*p)


def patchify_rgb(images, grid_thw, patch_size=16):
    h, w = image_grid(grid_thw)
    b, c, hh, ww = images.shape
    p = patch_size
    if (c, hh, ww) != (3, h*p, w*p):
        raise ValueError("RGB image does not match patch grid")
    return images.reshape(b, 3, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(b, h*w, p*p*3)


def processor_pixels_to_rgb(pixel_values, grid_thw, *, patch_size=16,
                            temporal_patch_size=2, merge_size=2,
                            mean=(.5, .5, .5), std=(.5, .5, .5)):
    """Exact de-packing and inverse normalization, not an encoder inversion.

    No second resize pipeline is used: target is the actual processed pixels.
    Used only by extraction/tests, never by the decoder forward method.
    """
    h, w = image_grid(grid_thw, merge_size)
    p, tp = patch_size, temporal_patch_size
    if pixel_values.shape != (h*w, 3*tp*p*p):
        raise ValueError("Unexpected processor pixel shape")
    x = pixel_values.reshape(h*w, 3, tp, p, p)
    if not torch.equal(x[:, :, :1].expand_as(x), x):
        raise ValueError("Video/nonidentical temporal slices are outside this image prototype")
    x = x[:, :, 0].permute(0, 2, 3, 1).reshape(1, h*w//(merge_size**2), merge_size**2, p*p*3)
    normalized = unpatchify(grouped_to_raster(x, grid_thw, merge_size), grid_thw, p)
    mu = normalized.new_tensor(mean).reshape(1, 3, 1, 1)
    sigma = normalized.new_tensor(std).reshape(1, 3, 1, 1)
    return normalized*sigma + mu
