"""Tests for per-tube augmentation transforms."""

import torch

from temporal_model.train.augment import (
    PhotometricTubeTransform,
    SpatialTubeTransform,
    TemporalTubeTransform,
    build_tube_augment,
)

_DEFAULT_CFG = {
    "enabled": True,
    "spatial": {
        "flip_prob": 0.5,
        "rotation_deg": 5.0,
        "scale_range": [0.9, 1.1],
        "translate_frac": 0.05,
    },
    "photometric": {
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2],
        "saturation_range": [0.8, 1.2],
    },
    "temporal": {
        "subseq_prob": 0.5,
        "subseq_min_len": 4,
        "stride_prob": 0.25,
        "frame_drop_prob": 0.15,
        "min_valid_after_drop": 4,
    },
}


def _make_item(t: int = 5, n_valid: int | None = None) -> dict:
    """Return a dict with T x 3 x 224 x 224 patches and a mask."""
    n_valid = t if n_valid is None else n_valid
    patches = torch.zeros(t, 3, 224, 224, dtype=torch.float32)
    # Put a distinct asymmetric pattern in each valid frame so flips are testable
    for i in range(n_valid):
        patches[i, :, :, :112] = 0.7  # bright left half
        patches[i, :, :, 112:] = 0.3  # darker right half
        patches[i, 0, i * 5, :] = 1.0  # red row tag that differs per frame
    mask = torch.zeros(t, dtype=torch.bool)
    mask[:n_valid] = True
    return {"patches": patches, "mask": mask}


def test_spatial_identity_preserves_input():
    """Ranges collapsed to neutral must return input unchanged."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    before = item["patches"].clone()
    t = SpatialTubeTransform(
        flip_prob=0.0,
        rotation_deg=0.0,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
    )
    out = t(item)
    assert torch.equal(out["patches"], before)


def test_spatial_flip_applied_identically_across_frames():
    """With flip_prob=1.0 every frame must be horizontally flipped, and
    the pre/post relationship is identical per frame (same flip decision)."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    before = item["patches"].clone()
    t = SpatialTubeTransform(
        flip_prob=1.0,
        rotation_deg=0.0,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
    )
    out = t(item)
    for i in range(4):
        assert torch.equal(out["patches"][i], torch.flip(before[i], dims=[-1]))


def test_spatial_affine_shape_preserved():
    """Affine ops keep the tensor shape."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    t = SpatialTubeTransform(
        flip_prob=0.5,
        rotation_deg=5.0,
        scale_range=(0.9, 1.1),
        translate_frac=0.05,
    )
    out = t(item)
    assert out["patches"].shape == (4, 3, 224, 224)
    assert out["mask"].shape == (4,)


def test_spatial_affine_applied_same_per_frame():
    """Affine params must be sampled once per tube, not per frame.

    Stack T identical frames, apply a non-trivial affine, assert every
    output frame equals frame 0. If the transform resampled angle/scale/
    translate per frame, the outputs would differ.
    """
    torch.manual_seed(0)
    # Build a tube where every frame is the same asymmetric pattern.
    one_frame = torch.zeros(3, 224, 224, dtype=torch.float32)
    one_frame[:, :, :112] = 0.7
    one_frame[:, :, 112:] = 0.3
    one_frame[0, 30, :] = 1.0  # single tag row
    patches = one_frame.unsqueeze(0).expand(5, -1, -1, -1).clone()
    item = {"patches": patches, "mask": torch.ones(5, dtype=torch.bool)}

    t = SpatialTubeTransform(
        flip_prob=0.0,
        rotation_deg=10.0,
        scale_range=(0.85, 1.15),
        translate_frac=0.08,
    )
    out = t(item)
    for i in range(1, 5):
        assert torch.equal(out["patches"][i], out["patches"][0]), (
            f"frame {i} differs from frame 0 — affine params were not "
            "shared across the tube"
        )


def test_photometric_identity_preserves_input():
    """Factors collapsed to 1.0 → output equals input."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    before = item["patches"].clone()
    t = PhotometricTubeTransform(
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
    )
    out = t(item)
    assert torch.allclose(out["patches"], before, atol=1e-6)


def test_photometric_same_factor_across_frames():
    """Brightness/contrast/saturation factors are sampled once per tube;
    the inter-frame difference pattern is preserved."""
    torch.manual_seed(0)
    item = _make_item(t=3)
    before = item["patches"].clone()
    # Compute pre-diffs between frames
    pre_diff_01 = before[0] - before[1]
    pre_diff_12 = before[1] - before[2]

    t = PhotometricTubeTransform(
        brightness_range=(0.8, 1.2),
        contrast_range=(1.0, 1.0),  # isolate: brightness-only
        saturation_range=(1.0, 1.0),
    )
    out = t(item)
    # Brightness is a per-pixel multiplicative factor; same factor per frame
    # means diffs scale by the same factor. So the ratio is preserved.
    post_diff_01 = out["patches"][0] - out["patches"][1]
    post_diff_12 = out["patches"][1] - out["patches"][2]
    # Avoid division-by-zero regions — use intersection so neither denominator is zero
    nz = (pre_diff_01.abs() > 1e-3) & (pre_diff_12.abs() > 1e-3)
    if nz.any():
        ratios_01 = post_diff_01[nz] / pre_diff_01[nz]
        ratios_12 = post_diff_12[nz] / pre_diff_12[nz]
        # All ratios should be the same single brightness factor
        assert torch.allclose(ratios_01, ratios_01[0].expand_as(ratios_01), atol=1e-4)
        assert torch.allclose(ratios_12, ratios_12[0].expand_as(ratios_12), atol=1e-4)


def _make_padded_item(t: int, n_valid: int) -> dict:
    """Tube with `n_valid` valid frames tagged by scalar value and rest padded."""
    patches = torch.zeros(t, 3, 224, 224, dtype=torch.float32)
    for i in range(n_valid):
        patches[i] = float(i + 1) / 100.0  # distinguishable per frame
    mask = torch.zeros(t, dtype=torch.bool)
    mask[:n_valid] = True
    return {"patches": patches, "mask": mask}


def test_temporal_identity_returns_input_unchanged():
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=10)
    before_patches = item["patches"].clone()
    before_mask = item["mask"].clone()
    t = TemporalTubeTransform(
        subseq_prob=0.0,
        subseq_min_len=4,
        stride_prob=0.0,
        frame_drop_prob=0.0,
        min_valid_after_drop=4,
    )
    out = t(item)
    assert torch.equal(out["patches"], before_patches)
    assert torch.equal(out["mask"], before_mask)


def test_temporal_mask_prefix_invariant_always_holds():
    """After any temporal transform the valid frames must occupy [0..k-1]."""
    for seed in range(30):
        torch.manual_seed(seed)
        item = _make_padded_item(t=20, n_valid=12)
        t = TemporalTubeTransform(
            subseq_prob=0.5,
            subseq_min_len=4,
            stride_prob=0.25,
            frame_drop_prob=0.15,
            min_valid_after_drop=4,
        )
        out = t(item)
        k = int(out["mask"].sum())
        assert out["mask"][:k].all(), f"seed={seed}: non-contiguous True prefix"
        assert not out["mask"][k:].any(), f"seed={seed}: stray True after prefix"
        # Padded positions are zeros
        if k < 20:
            assert out["patches"][k:].abs().sum() == 0.0


def test_temporal_subsequence_contiguous_slice():
    """With subseq_prob=1, the valid frames of the output are a contiguous
    slice of the valid frames of the input."""
    torch.manual_seed(42)
    item = _make_padded_item(t=20, n_valid=10)
    # Original per-frame tags: 0.01, 0.02, ..., 0.10 (on first pixel)
    original_tags = [float(item["patches"][i, 0, 0, 0].item()) for i in range(10)]
    t = TemporalTubeTransform(
        subseq_prob=1.0,
        subseq_min_len=4,
        stride_prob=0.0,
        frame_drop_prob=0.0,
        min_valid_after_drop=4,
    )
    out = t(item)
    k = int(out["mask"].sum())
    out_tags = [float(out["patches"][i, 0, 0, 0].item()) for i in range(k)]
    # Find out_tags as a contiguous slice of original_tags
    assert k >= 4
    for start in range(10 - k + 1):
        if original_tags[start : start + k] == out_tags:
            break
    else:
        raise AssertionError(
            f"out_tags {out_tags} is not a contiguous slice of {original_tags}"
        )


def test_temporal_stride_halves_length():
    """stride_prob=1 means every second frame kept: length ~= ceil(n/2)."""
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=10)
    t = TemporalTubeTransform(
        subseq_prob=0.0,
        subseq_min_len=4,
        stride_prob=1.0,
        frame_drop_prob=0.0,
        min_valid_after_drop=2,
    )
    out = t(item)
    k = int(out["mask"].sum())
    assert k == 5  # ceil(10/2)


def test_temporal_frame_drop_respects_floor():
    """Very aggressive drop prob still leaves at least min_valid_after_drop."""
    for seed in range(20):
        torch.manual_seed(seed)
        item = _make_padded_item(t=20, n_valid=10)
        t = TemporalTubeTransform(
            subseq_prob=0.0,
            subseq_min_len=4,
            stride_prob=0.0,
            frame_drop_prob=0.99,
            min_valid_after_drop=4,
        )
        out = t(item)
        k = int(out["mask"].sum())
        assert k >= 4, f"seed={seed}: dropped below floor ({k})"


def test_temporal_compacts_dropped_frames_to_zero_prefix():
    """After drop, remaining valid patches are at positions [0..k-1], not
    scattered with zeros in between."""
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=6)
    # Tag each valid frame uniquely in its top-left pixel
    for i in range(6):
        item["patches"][i, 0, 0, 0] = float(i + 1)
    t = TemporalTubeTransform(
        subseq_prob=0.0,
        subseq_min_len=4,
        stride_prob=0.0,
        frame_drop_prob=0.5,
        min_valid_after_drop=3,
    )
    out = t(item)
    k = int(out["mask"].sum())
    # Every valid output frame must carry a nonzero tag (i.e. not a zero pad).
    for i in range(k):
        assert out["patches"][i, 0, 0, 0].item() > 0.5


def test_val_transform_is_normalize_only_and_deterministic():
    """train=False must skip aug and just apply ImageNet normalize."""
    torch.manual_seed(0)
    item_a = _make_padded_item(t=20, n_valid=5)
    item_b = _make_padded_item(t=20, n_valid=5)
    # Mutate item_b in place to prove the transform is not sampling randomness
    transform = build_tube_augment(_DEFAULT_CFG, train=False)
    out_a = transform(item_a)
    out_b = transform(item_b)
    assert torch.allclose(out_a["patches"][:5], out_b["patches"][:5])

    # Normalization actually happened: raw 0.01 != normalized 0.01
    raw = 0.01
    assert not torch.allclose(
        out_a["patches"][0], torch.full((3, 224, 224), raw), atol=1e-3
    )


def test_val_transform_does_not_mutate_mask_or_length():
    item = _make_padded_item(t=20, n_valid=5)
    before_mask = item["mask"].clone()
    transform = build_tube_augment(_DEFAULT_CFG, train=False)
    out = transform(item)
    assert torch.equal(out["mask"], before_mask)
    assert out["patches"].shape == (20, 3, 224, 224)


def test_train_transform_applies_pipeline_and_normalizes():
    """train=True composes spatial -> photometric -> temporal -> normalize."""
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=10)
    transform = build_tube_augment(_DEFAULT_CFG, train=True)
    out = transform(item)
    # Shape preserved
    assert out["patches"].shape == (20, 3, 224, 224)
    # Valid mask still contiguous prefix
    k = int(out["mask"].sum())
    assert out["mask"][:k].all()
    assert not out["mask"][k:].any()
    # Output is normalized: valid frames should have values outside [0, 1].
    # ImageNet normalize on [0, 1] input yields negative values for R channel.
    assert out["patches"][:k].min() < 0.0 or out["patches"][:k].max() > 1.0


def test_disabled_config_skips_aug_in_train_mode():
    """enabled=False + train=True must behave like val: normalize only."""
    torch.manual_seed(0)
    cfg_off = dict(_DEFAULT_CFG)
    cfg_off["enabled"] = False
    item_a = _make_padded_item(t=20, n_valid=5)
    item_b = _make_padded_item(t=20, n_valid=5)
    t_train = build_tube_augment(cfg_off, train=True)
    t_val = build_tube_augment(cfg_off, train=False)
    out_a = t_train(item_a)
    out_b = t_val(item_b)
    assert torch.allclose(out_a["patches"], out_b["patches"])


def test_train_transform_reproducible_with_fixed_seed():
    torch.manual_seed(7)
    item1 = _make_padded_item(t=20, n_valid=10)
    out1 = build_tube_augment(_DEFAULT_CFG, train=True)(item1)

    torch.manual_seed(7)
    item2 = _make_padded_item(t=20, n_valid=10)
    out2 = build_tube_augment(_DEFAULT_CFG, train=True)(item2)

    assert torch.equal(out1["patches"], out2["patches"])
    assert torch.equal(out1["mask"], out2["mask"])
