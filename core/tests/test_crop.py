"""Tests for core.crop geometry."""

import numpy as np
import pytest

from temporal_model.core.crop import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)


def test_expand_bbox_scales_w_and_h_by_factor():
    cx, cy, w, h = expand_bbox(0.5, 0.5, 0.1, 0.2, factor=1.5)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.15)
    assert h == pytest.approx(0.30)


def test_expand_bbox_factor_one_is_identity():
    assert expand_bbox(0.3, 0.7, 0.04, 0.06, factor=1.0) == (0.3, 0.7, 0.04, 0.06)


def test_norm_bbox_to_pixel_square_returns_square_inside_bounds():
    box = norm_bbox_to_pixel_square(0.5, 0.5, 0.1, 0.2, img_w=1000, img_h=800)
    x0, y0, x1, y1 = box
    side = x1 - x0
    assert side == y1 - y0
    assert side == 160
    assert (x0 + x1) // 2 == 500
    assert (y0 + y1) // 2 == 400


def test_norm_bbox_to_pixel_square_clips_at_left_edge():
    box = norm_bbox_to_pixel_square(0.02, 0.5, 0.1, 0.1, img_w=1000, img_h=1000)
    x0, y0, x1, y1 = box
    assert x0 >= 0
    assert y0 >= 0
    assert x1 <= 1000
    assert y1 <= 1000


def test_norm_bbox_to_pixel_square_returns_integer_coords():
    box = norm_bbox_to_pixel_square(0.333, 0.777, 0.111, 0.222, img_w=1280, img_h=720)
    for v in box:
        assert isinstance(v, int)


def _solid_image(w: int, h: int, color: tuple[int, int, int]) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    return arr


def test_crop_and_resize_returns_uint8_rgb_at_target_size():
    img = _solid_image(800, 600, (255, 0, 0))
    patch = crop_and_resize(img, (100, 100, 300, 300), patch_size=224)
    assert patch.shape == (224, 224, 3)
    assert patch.dtype == np.uint8


def test_crop_and_resize_pads_non_square_crop_with_zeros():
    img = _solid_image(800, 600, (255, 255, 255))
    patch = crop_and_resize(img, (300, 0, 500, 100), patch_size=224)
    assert patch.shape == (224, 224, 3)
    assert patch[0, 100, :].sum() == 0
    assert patch[223, 100, :].sum() == 0
