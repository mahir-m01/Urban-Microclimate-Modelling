"""Tests for the S10 deep-learning data pipeline (urbanwind.dl.cache and urbanwind.dl.datasets).

Everything here runs on small hand-made fields, so no raw data is needed. The last test also
checks real cached cases against the shared loader and skips when the cache has not been built.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # PyTorch has no wheels for Intel Macs; CI and Apple silicon have it

from urbanwind.data import EXCLUDE, case_table
from urbanwind.dl.cache import (CACHE_DIR, CACHE_VERSION, H_SCALE, SIN_CHANNEL, _splits_hash, build_arrays,
                                build_case, canonical_transform, dev_cases, from_canonical, load_cached,
                                make_inputs, resample_to_1m, to_canonical, vector_to_canonical, with_inputs,
                                write_case)
from urbanwind.dl.datasets import (MARGIN, PatchDataset, fill_buildings, mirror_patch, padded_size, predict_arrays,
                                   predict_canonical, predict_native, upsample2_periodic, wrap_window)
from urbanwind.features import compute_periodic_distance_transform
from urbanwind.splits import load_splits

# every direction in the development set, plus a sweep of whole degrees
DIRECTIONS = [0, 15, 22.5, 30, 45, 60, 75, 90, 270, 285, 292.5, 300, 315, 330, 337.5, 345, *range(0, 360, 7), 359.9,
              -1e-14]  # -1e-14 % 360 is exactly 360.0 in floating point
EIGHT = [(k, t) for k in range(4) for t in (False, True)]


# ---------------------------------------------------------------------------
# Canonical wind frame
# ---------------------------------------------------------------------------

def test_every_transform_is_undone_exactly():
    rng = np.random.default_rng(0)
    field, stack, square = rng.random((5, 7)), rng.random((3, 5, 7)), rng.random((6, 6))
    for k, t in EIGHT:
        assert np.array_equal(from_canonical(to_canonical(field, k, t), k, t), field)
        assert np.array_equal(from_canonical(to_canonical(stack, k, t), k, t), stack)
    # the eight (k, transposed) pairs really are the eight different symmetries of a square
    assert len({to_canonical(square, k, t).tobytes() for k, t in EIGHT}) == 8


@pytest.mark.parametrize("theta", DIRECTIONS)
def test_wind_vector_lands_at_the_remaining_angle(theta):
    k, t, angle = canonical_transform(theta)
    assert 0.0 <= angle <= 45.0
    rad = np.deg2rad(theta)
    u, v = vector_to_canonical(np.array([[np.cos(rad)]]), np.array([[np.sin(rad)]]), k, t)
    assert np.allclose([u[0, 0], v[0, 0]], [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])


@pytest.mark.parametrize("theta", DIRECTIONS)
def test_field_rising_along_the_wind_still_rises_along_the_wind(theta):
    # f = distance travelled along the wind. After the array is turned, it must equal the
    # distance along the remaining angle in the new row/column coordinates, up to a constant.
    ny, nx = 6, 10
    rows, cols = np.mgrid[0:ny, 0:nx]
    rad = np.deg2rad(theta)
    f = cols * np.cos(rad) + rows * np.sin(rad)
    k, t, angle = canonical_transform(theta)
    g = to_canonical(f, k, t)
    new_rows, new_cols = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    a = np.deg2rad(angle)
    offset = g - (new_cols * np.cos(a) + new_rows * np.sin(a))
    assert np.allclose(offset, offset.flat[0])


# ---------------------------------------------------------------------------
# 1 m resampling
# ---------------------------------------------------------------------------

def test_resampling_matches_hand_example():
    nan = np.nan
    H = np.array([[0, 0, 3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [5, 0, 0, 0, 0, 0]], dtype=float)
    U = np.array([[1, 2, nan, 4, 1, 1],
                  [3, 4, 5, 6, 1, nan],  # the NaN on open ground is a zero-speed cell
                  [1, 1, 2, 2, 0.5, 1.5],
                  [nan, 1, 2, 2, 2.5, 3.5]])
    H1, out = resample_to_1m(H, {"U": U}, 0.5)
    assert np.array_equal(H1, [[0, 3, 0], [5, 0, 0]])
    assert np.array_equal(out["U"], [[2.5, nan, nan], [nan, 2.0, 2.0]], equal_nan=True)


def test_resampling_passes_1m_through_and_refuses_odd_grids():
    H = np.zeros((4, 4))
    assert resample_to_1m(H, {}, 1.0)[0] is H
    with pytest.raises(ValueError):
        resample_to_1m(np.zeros((5, 4)), {}, 0.5)
    with pytest.raises(ValueError):
        resample_to_1m(H, {}, 2.0)


# ---------------------------------------------------------------------------
# Test set and case list
# ---------------------------------------------------------------------------

def test_no_test_group_or_excluded_case_can_enter_the_cache():
    dev = dev_cases()
    splits = load_splits()
    assert len(dev) == 482
    assert dev["subset"].value_counts().to_dict() == {"realistic": 292, "idealized": 190}
    assert not set(dev["split_group"]) & set(splits["test_groups"])
    assert not set(dev.index) & set(EXCLUDE)
    assert set(dev["fold"]) == set(range(splits["n_folds"]))
    test_case = case_table().index[case_table()["split_group"].isin(splits["test_groups"])][0]
    for case in (test_case, next(iter(EXCLUDE))):
        with pytest.raises(ValueError, match="not a development case"):
            build_case(case)  # refused before any file is opened


# ---------------------------------------------------------------------------
# Synthetic cases written to a temporary cache
# ---------------------------------------------------------------------------

def fake_case(rng, ny, nx, dx, wind_dir):
    """A dict shaped like load_case output: random buildings, wind fields NaN off the valid cells."""
    H = np.where(rng.random((ny, nx)) < 0.2, rng.integers(1, 30, (ny, nx)), 0).astype(float)
    valid = (H == 0) & (rng.random((ny, nx)) > 0.02)  # a few zero-speed cells too
    fields = {name: np.where(valid, rng.random((ny, nx)) + 0.5, np.nan) for name in ("U", "u", "v", "TKE")}
    return {"H": H, "dx": dx, "wind_dir_deg": wind_dir, **fields}


def write_fake(cache_dir, name, c):
    meta = {"u_tau": 0.2, "subset": "idealized", "split_group": "fake", "status": "ok", "fold": 0,
            "version": CACHE_VERSION, "splits_hash": _splits_hash()}
    write_case(name, build_arrays(c) | meta, cache_dir)


@pytest.fixture
def fake_cache(tmp_path):
    rng = np.random.default_rng(1)
    cases = {"small_05m": fake_case(rng, 80, 120, 0.5, 250.0),  # 40 x 60 at 1 m: smaller than a patch
             "city_1m": fake_case(rng, 300, 280, 1.0, 120.0)}
    for name, c in cases.items():
        write_fake(tmp_path, name, c)
    return tmp_path, cases


def test_cached_case_turns_back_into_the_loader_fields(fake_cache):
    cache_dir, cases = fake_cache
    native = cases["city_1m"]
    cached = load_cached("city_1m", cache_dir)
    back = from_canonical(cached["U"], cached["k"], cached["transposed"]).astype(float)
    assert np.array_equal(np.isnan(back), np.isnan(native["U"]))
    assert np.allclose(back, native["U"], atol=2e-3, equal_nan=True)  # float16 storage
    assert np.isfinite(cached["y"]).all() and cached["x"].shape == (5, *cached["valid"].shape)
    # the native-grid truth the evaluator scores against is stored exactly, for both grid spacings
    for name in ("city_1m", "small_05m"):
        assert np.array_equal(load_cached(name, cache_dir)["U_native"], cases[name]["U"].astype(np.float32),
                              equal_nan=True)


def test_mirror_equals_the_inputs_of_the_mirrored_case():
    rng = np.random.default_rng(2)
    H = np.where(rng.random((40, 50)) < 0.2, 10.0, 0.0)
    dist = compute_periodic_distance_transform(H, dx=1.0)[0]
    y, mask = rng.random((40, 50)), H == 0
    x_m, y_m, mask_m = mirror_patch(make_inputs(H, dist, 30.0), y, mask)
    flipped = H[::-1]
    expected = make_inputs(flipped, compute_periodic_distance_transform(flipped, dx=1.0)[0], -30.0)
    assert np.allclose(x_m, expected, atol=1e-6)
    assert x_m[SIN_CHANNEL].max() < 0
    assert np.array_equal(y_m, y[::-1]) and np.array_equal(mask_m, mask[::-1])


def test_periodic_bilinear_upsampling():
    rng = np.random.default_rng(3)
    a = rng.random((6, 10))
    tiled = torch.nn.functional.interpolate(torch.from_numpy(np.tile(a, (3, 3)))[None, None], scale_factor=2,
                                            mode="bilinear", align_corners=False)[0, 0].numpy()
    assert np.allclose(upsample2_periodic(a), tiled[12:24, 20:40])  # the middle tile: no edge effects
    assert np.allclose(upsample2_periodic(np.full((4, 6), 2.5)), 2.5)


def test_wrap_window_wraps_and_tiles():
    a = np.arange(35).reshape(5, 7)
    assert np.array_equal(wrap_window(a, 3, 5, 4, 6), np.roll(a, (-3, -5), axis=(0, 1))[:4, :6])
    assert np.array_equal(wrap_window(a, -2, -1, 4, 6), np.roll(a, (2, 1), axis=(0, 1))[:4, :6])
    big = np.tile(np.roll(a, (-2, -1), axis=(0, 1)), (3, 3))[:12, :16]
    assert np.array_equal(wrap_window(a, 2, 1, 12, 16), big)


def test_patch_dataset_items(fake_cache):
    cache_dir, _ = fake_cache
    plain = PatchDataset(["small_05m", "city_1m"], patch=64, mirror=False, seed=3, border=8, cache_dir=cache_dir)
    mirrored = PatchDataset(["small_05m", "city_1m"], patch=64, mirror=True, seed=3, border=8, cache_dir=cache_dir)
    assert len(plain) == int(np.ceil((40 * 60 + 300 * 280) / 64**2))
    flips = []
    for i in range(30):
        x, y, mask = plain[i]
        assert (x.shape, y.shape, mask.shape) == ((5, 64, 64), (1, 64, 64), (1, 64, 64))
        assert x.dtype == y.dtype == torch.float32 and mask.dtype == torch.bool
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        assert not mask[0, :8].any() and not mask[0, -8:].any() and not mask[0, :, :8].any() and not mask[0, :, -8:].any()
        assert not (mask[0] & (x[1] > 0)).any()  # never a valid cell on a building
        xm, ym, mm = mirrored[i]
        same = torch.equal(xm, x) and torch.equal(ym, y) and torch.equal(mm, mask)
        fx, fy, fm = mirror_patch(x.numpy(), y[0].numpy(), mask[0].numpy())
        flipped = np.array_equal(xm.numpy(), fx) and np.array_equal(ym[0].numpy(), fy) and np.array_equal(mm[0].numpy(), fm)
        assert same or flipped
        flips.append(flipped and not same)
    assert 0 < sum(flips) < 30
    # reproducible for a given (seed, epoch, index), and fresh patches in a new epoch
    first = plain[5][0]
    assert torch.equal(first, plain[5][0])
    plain.epoch = 1
    assert not torch.equal(first, plain[5][0])
    # the order the cases are given in does not change the patches
    swapped = PatchDataset(["city_1m", "small_05m"], patch=64, mirror=False, seed=3, border=8, cache_dir=cache_dir)
    plain.epoch = 0
    assert all(torch.equal(swapped[i][0], plain[i][0]) for i in range(20))


def height_model() -> torch.nn.Module:
    """A 1x1 convolution that returns the height channel times H_SCALE, i.e. the building heights."""
    model = torch.nn.Conv2d(5, 1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0] = H_SCALE
    return model


def open_model() -> torch.nn.Module:
    """A 1x1 convolution that returns 1 on open ground and 0 on buildings."""
    model = torch.nn.Conv2d(5, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 1] = -1.0
        model.bias.fill_(1.0)
    return model


def test_predict_native_returns_the_native_grid(fake_cache):
    cache_dir, cases = fake_cache
    model = height_model()
    H_city = predict_native(model, "city_1m", cache_dir, fill=False)
    assert np.array_equal(H_city, cases["city_1m"]["H"])
    H_small = predict_native(model, "small_05m", cache_dir, fill=False, upsample="copy")
    blocks = cases["small_05m"]["H"].reshape(40, 2, 60, 2).max(axis=(1, 3))
    assert np.array_equal(H_small, blocks.repeat(2, axis=0).repeat(2, axis=1))
    # the default spreads the 1 m field over the 0.5 m grid by periodic bilinear interpolation
    smooth = predict_native(model, "small_05m", cache_dir, fill=False)
    assert np.allclose(smooth, upsample2_periodic(blocks), atol=1e-4)
    # open_model gives 0 on 1 m blocks that hold a building; some of their 0.5 m sub-cells are scored
    raw = predict_native(open_model(), "small_05m", cache_dir, fill=False)
    scored = np.isfinite(cases["small_05m"]["U"])
    assert (raw[scored] == 0).any()
    filled = predict_native(open_model(), "small_05m", cache_dir)
    assert (filled == 1).all()  # every cell now carries the prediction of an open cell
    # a case built in memory (how final scoring handles test cases) gives the same answer as the cache
    in_memory = with_inputs(build_arrays(cases["small_05m"]))
    assert np.array_equal(predict_arrays(open_model(), in_memory), filled)
    assert np.array_equal(predict_arrays(height_model(), in_memory, fill=False, upsample="copy"), H_small)
    for n in (40, 60, 256, 280, 1150, 1152):
        before, total = padded_size(n)
        if n % 16 == 0:  # already periodic for the network: no padding at all
            assert (before, total) == (0, n)
        else:
            assert total % 16 == 0 and before >= MARGIN and total - n - before >= MARGIN


# ---------------------------------------------------------------------------
# Real cache (skips until build_cache.py has run)
# ---------------------------------------------------------------------------

# a 0.5 m case at 45 degrees (resampled only), a 90 degree city (turned) and a 60 degree city (transposed)
REAL = ["UA25_d45", "AU-Ade-V1_d90", "CN-BE-V1_d150"]


@pytest.mark.skipif(not all((CACHE_DIR / f"{case}.npz").exists() for case in REAL), reason="S10 cache not built")
def test_real_cached_case_matches_the_loader():
    from urbanwind.data import load_case
    for case in REAL:
        c, cached = load_case(case), load_cached(case)
        H, fields = resample_to_1m(c["H"], {"U": c["U"]}, c["dx"])
        assert np.array_equal(from_canonical(cached["H"], cached["k"], cached["transposed"]), H)
        back = from_canonical(cached["U"], cached["k"], cached["transposed"]).astype(float)
        assert np.array_equal(np.isnan(back), np.isnan(fields["U"]))
        assert np.nanmax(np.abs(back - fields["U"])) < 1e-2
        # the prediction lands on the native grid of the raw files, cell by cell
        expected = H if np.isclose(c["dx"], 1.0) else H.repeat(2, axis=0).repeat(2, axis=1)
        assert np.array_equal(predict_native(height_model(), case, fill=False, upsample="copy"), expected)
        assert (predict_native(open_model(), case) == 1).all()


def periodic_conv() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Conv2d(5, 1, 3, padding=1, padding_mode="circular", bias=False).eval()


@pytest.mark.parametrize("shape", [(40, 60), (96, 128), (150, 230)])
def test_whole_and_tiled_prediction_equal_the_periodic_answer(shape):
    # a 3x3 circular convolution has an exact answer on a periodic domain: convolve the domain itself
    rng = np.random.default_rng(4)
    c = {"x": rng.random((5, *shape)).astype(np.float32), "angle_deg": 10.0}
    model = periodic_conv()
    with torch.no_grad():
        truth = model(torch.from_numpy(c["x"])[None])[0, 0].numpy()
    whole = predict_canonical(model, c, "cpu", fill=False)
    tiled = predict_canonical(model, c, "cpu", fill=False, window=64, border=8)
    assert np.allclose(whole, truth, atol=1e-5) and np.allclose(tiled, truth, atol=1e-5)


def test_symmetry_averaging_keeps_a_symmetric_answer():
    # the height model just returns the heights, so every symmetric copy, turned back, gives the same field
    rng = np.random.default_rng(5)
    H = np.where(rng.random((64, 80)) < 0.2, 9.0, 0.0)
    for angle in (45.0, 20.0):  # 45 degrees adds the transpose to the mirror
        c = {"x": make_inputs(H, compute_periodic_distance_transform(H, dx=1.0)[0], angle), "angle_deg": angle}
        plain = predict_canonical(height_model(), c, "cpu", fill=False)
        averaged = predict_canonical(height_model(), c, "cpu", fill=False, mirror_tta=True)
        assert np.allclose(plain, H) and np.allclose(averaged, H)


def test_fill_reaches_across_the_periodic_edge():
    pred = np.arange(30, dtype=float).reshape(5, 6)
    buildings = np.zeros((5, 6), bool)
    buildings[:, 0:3] = True  # a wall three columns wide: column 0's nearest open cell is column 5, across the edge
    filled = fill_buildings(pred, buildings)
    assert np.array_equal(filled[:, 0], pred[:, 5]) and np.array_equal(filled[:, 3:], pred[:, 3:])


def test_45_degree_patches_are_sometimes_transposed(tmp_path):
    rng = np.random.default_rng(6)
    write_fake(tmp_path, "at45", fake_case(rng, 120, 100, 1.0, 45.0))
    write_fake(tmp_path, "at30", fake_case(rng, 120, 100, 1.0, 30.0))
    plain = PatchDataset(["at45"], patch=32, mirror=False, border=0, cache_dir=tmp_path, transpose45=False)
    turned = PatchDataset(["at45"], patch=32, mirror=False, border=0, cache_dir=tmp_path)
    kinds = []
    for i in range(20):
        a, b = plain[i][0], turned[i][0]
        kinds.append("same" if torch.equal(a, b) else "T" if torch.equal(a.transpose(-2, -1), b) else "other")
    assert set(kinds) == {"same", "T"}
    other = PatchDataset(["at30"], patch=32, mirror=False, border=0, cache_dir=tmp_path)
    other_plain = PatchDataset(["at30"], patch=32, mirror=False, border=0, cache_dir=tmp_path, transpose45=False)
    assert all(torch.equal(other[i][0], other_plain[i][0]) for i in range(10))  # never at other angles


def test_45_degree_copies_are_all_valid_symmetries(tmp_path):
    # with the mirror on as well, a 45-degree patch may be transposed and then mirrored, never the other
    # way round (that would point the wind at 135 degrees); the sine channel must match the rows' flip
    rng = np.random.default_rng(7)
    write_fake(tmp_path, "at45", fake_case(rng, 120, 100, 1.0, 45.0))
    base = PatchDataset(["at45"], patch=32, mirror=False, border=0, cache_dir=tmp_path, transpose45=False)
    full = PatchDataset(["at45"], patch=32, mirror=True, border=0, cache_dir=tmp_path)
    seen = set()
    for i in range(40):
        p0, p1 = base[i][0].numpy(), full[i][0].numpy()
        sin_sign = np.sign(p1[SIN_CHANNEL, 0, 0])
        valid = {"plain": (p0, 1), "T": (np.swapaxes(p0, -2, -1), 1), "M": (p0[:, ::-1], -1),
                 "TM": (np.swapaxes(p0, -2, -1)[:, ::-1], -1)}
        match = [k for k, (ref, sign) in valid.items()
                 if np.allclose(p1[[0, 1, 2]], ref[[0, 1, 2]]) and sin_sign == sign]
        assert match, f"patch {i} is not a valid symmetric copy"
        seen.add(match[0])
    assert seen == {"plain", "T", "M", "TM"}
