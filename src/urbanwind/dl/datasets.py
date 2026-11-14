"""
datasets.py: training patches and full-domain prediction on top of the S10 cache.

Public API
----------
wrap_window(a, r0, c0, h, w)  : an h×w window of a periodic field, wrapping across its edges
mirror_patch(x, y, mask)      : mirror a patch across the wind axis
PatchDataset(cases, ...)      : random 256×256 training patches as (x, y, mask) tensors
upsample2_periodic(a)         : periodic bilinear 2x upsampling (1 m prediction to a 0.5 m grid)
predict_canonical(model, c)   : U/u_tau on the 1 m canonical grid (used for validation)
predict_arrays(model, arrays) : U/u_tau on the native grid for one case held in memory
predict_native(model, case)   : the same for a cached development case
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from urbanwind.dl.cache import (BUILDING_CHANNEL, CACHE_DIR, FINE_DX, SIN_CHANNEL, from_canonical, load_cached,
                                make_inputs, read_cached)

PATCH = 256
BORDER = 16
MARGIN = 128  # periodic context around a whole domain before prediction; the U-Net sees about 126 cells
MULTIPLE = 16  # four 2x poolings in the U-Net need sides divisible by 16; a model may ask for more


def wrap_window(a: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """The h×w window of a periodic field whose top-left cell is (r0, c0), over the last two axes.

    The indices wrap around the domain, so a window that runs off one edge continues from the
    opposite edge, and a window larger than the domain simply tiles it.
    """
    ny, nx = a.shape[-2:]
    rows = (r0 + np.arange(h)) % ny
    cols = (c0 + np.arange(w)) % nx
    return a[..., rows[:, None], cols[None, :]]


def mirror_patch(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror a patch across the wind axis: flip the rows and flip the sign of the wind angle.

    Flipping the rows sends a wind angle a to -a, so only the sine channel changes sign. Without
    Coriolis forcing the mirrored case is as valid a flow as the original. This is exact for a
    scalar target such as U; a target with a v component would also need its sign flipped.
    """
    x = x[:, ::-1].copy()
    x[SIN_CHANNEL] *= -1.0
    return x, y[::-1].copy(), mask[::-1].copy()


class PatchDataset(Dataset):
    """Random square training patches cut from whole cached cases.

    Each item draws a case with probability proportional to the square root of its open area
    (so large cities do not drown out the small idealized layouts), a random top-left cell, and
    with ``mirror`` a coin flip for the mirror. Items depend only on ``(seed, epoch, index)``, so
    a run is reproducible and stays correct with several DataLoader workers. Set ``epoch`` before
    each epoch to get fresh patches; with ``persistent_workers=True`` the workers keep their own
    copy and never see the new epoch, so leave that option off.

    The network pads every patch circularly, which glues its top edge to its bottom edge even
    though they are not neighbours in the city. ``border`` cells along each side are therefore
    left out of the loss mask.

    Only the stored float16 height, wall distance and target are kept in memory (about 0.7 GB for
    all 482 cases); the five input channels are built for each patch.
    """

    def __init__(self, cases: list[str], patch: int = PATCH, mirror: bool = True, seed: int = 0,
                 border: int = BORDER, cache_dir: pathlib.Path = CACHE_DIR, length: int | None = None,
                 transpose45: bool = True):
        if not cases:
            raise ValueError("PatchDataset needs at least one case")
        if not 0 <= 2 * border < patch:
            raise ValueError(f"border {border} leaves nothing of a {patch} patch")
        self.patch, self.mirror, self.seed, self.border, self.epoch = patch, mirror, seed, border, 0
        self.transpose45 = transpose45
        self.cases = sorted(cases)  # the patches then do not depend on the order the cases came in
        self.items = [{"H": c["H"], "dist": c["dist"], "y": np.nan_to_num(c["U"], nan=0.0), "valid": c["valid"],
                       "angle_deg": c["angle_deg"]} for c in (read_cached(case, cache_dir) for case in self.cases)]
        open_area = np.array([item["valid"].sum() for item in self.items], dtype=float)
        self.weights = np.sqrt(open_area) / np.sqrt(open_area).sum()
        # by default one epoch holds as many patches as it takes to cover the cases' total area once;
        # training passes a fixed length instead, so every fold gets the same number of steps
        self.length = length or math.ceil(sum(item["valid"].size for item in self.items) / patch**2)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, index))
        item = self.items[rng.choice(len(self.items), p=self.weights)]
        ny, nx = item["valid"].shape
        r0, c0 = int(rng.integers(ny)), int(rng.integers(nx))
        H, dist, y, mask = (wrap_window(item[name], r0, c0, self.patch, self.patch) for name in ("H", "dist", "y", "valid"))
        x, y = make_inputs(H, dist, item["angle_deg"]), y.astype(np.float32)
        if self.border:
            b = self.border
            mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = False, False, False, False
        flip = self.mirror and rng.random() < 0.5
        # at exactly 45 degrees the transpose keeps the wind angle, so it is one more exact copy. It goes
        # before the mirror: transposing a mirrored case (wind at -45) would send the wind to 135 degrees
        if self.transpose45 and item["angle_deg"] == 45.0 and rng.random() < 0.5:
            x, y, mask = (np.ascontiguousarray(np.swapaxes(a, -2, -1)) for a in (x, y, mask))
        if flip:
            x, y, mask = mirror_patch(x, y, mask)
        return torch.from_numpy(x), torch.from_numpy(y[None]), torch.from_numpy(mask[None])


def padded_size(n: int, multiple: int = MULTIPLE, margin: int = MARGIN) -> tuple[int, int]:
    """(cells added before, total length) for one side of a whole domain.

    A side that the network's pooling already divides needs no padding: the circular padding then
    wraps the domain onto itself, which is exactly its periodicity. Any other side gets at least
    ``margin`` cells of its own periodic continuation on each end, rounded up to ``multiple``, so the
    seam at the edge of the input lies beyond the network's reach.
    """
    if n % multiple == 0:
        return 0, n
    total = math.ceil((n + 2 * margin) / multiple) * multiple
    return (total - n) // 2, total


def _forward(model, x: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(x))[None].to(device))[0, 0].float().cpu().numpy()


def predict_whole(model, x: np.ndarray, device, margin: int = MARGIN) -> np.ndarray:
    """One pass over the whole domain (padded only when its sides do not divide)."""
    _, ny, nx = x.shape
    multiple = getattr(model, "multiple", MULTIPLE)  # a 5-level U-Net needs sides divisible by 32
    top, h = padded_size(ny, multiple, margin)
    left, w = padded_size(nx, multiple, margin)
    return _forward(model, wrap_window(x, -top, -left, h, w), device)[top:top + ny, left:left + nx]


def predict_tiled(model, x: np.ndarray, device, window: int, border: int, batch: int = 8) -> np.ndarray:
    """Predict in windows of the training patch size and keep each window's centre.

    The network then always sees inputs of the size it was trained on, with the same normalisation
    statistics and the circular seam ``border`` cells away from every kept cell, as in training.
    """
    _, ny, nx = x.shape
    step = window - 2 * border
    starts = [(r, c) for r in range(-border, ny - border, step) for c in range(-border, nx - border, step)]
    out = np.empty((ny, nx), np.float32)
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        tiles = np.stack([wrap_window(x, r, c, window, window) for r, c in chunk])
        with torch.no_grad():
            pred = model(torch.from_numpy(tiles).to(device))[:, 0].float().cpu().numpy()
        for (r, c), tile in zip(chunk, pred):
            rows, cols = (r + border + np.arange(step)) % ny, (c + border + np.arange(step)) % nx
            out[np.ix_(rows, cols)] = tile[border:border + step, border:border + step]
    return out


def fill_buildings(pred: np.ndarray, buildings: np.ndarray) -> np.ndarray:
    """Give every building cell the value of the nearest open cell, across the periodic edges too."""
    ny, nx = pred.shape
    if not buildings.any() or buildings.all():
        return pred
    nearest = ndimage.distance_transform_edt(np.tile(buildings, (3, 3)), return_distances=False, return_indices=True)
    rows, cols = nearest[0][ny:2 * ny, nx:2 * nx] % ny, nearest[1][ny:2 * ny, nx:2 * nx] % nx
    return pred[rows, cols]


def _variants(x: np.ndarray, angle_deg: float, mirror_tta: bool) -> list[tuple[np.ndarray, Callable]]:
    """The exact symmetric copies of one case, each with the function that turns its prediction back."""
    out = [(x, lambda p: p)]
    if angle_deg == 45.0:  # the transpose keeps a 45 degree wind at 45 degrees
        out.append((np.swapaxes(x, -2, -1), lambda p: p.T))
    if mirror_tta:
        mirrored = []
        for xi, back in out:
            m = xi[:, ::-1].copy()
            m[SIN_CHANNEL] *= -1.0
            mirrored.append((m, lambda p, back=back: back(p[::-1])))
        out += mirrored
    return out


def predict_canonical(model: torch.nn.Module, c: dict[str, Any], device: torch.device | str | None = None,
                      fill: bool = True, mirror_tta: bool = False, window: int = 0, border: int = BORDER,
                      margin: int = MARGIN) -> np.ndarray:
    """U/u_tau for one whole case on its 1 m grid in the canonical frame (before turning back).

    ``c`` holds the arrays of one case with its inputs (``load_cached`` or ``with_inputs``). With
    ``window`` 0 the domain is predicted in one pass (``predict_whole``), otherwise in windows of that
    size (``predict_tiled``). With ``mirror_tta`` the prediction is averaged over the exact symmetric
    copies of the case: the mirror (rows flipped, angle negated) and, at 45 degrees, the transpose.
    With ``fill`` every cell that the input marks as a building takes the prediction of the nearest
    open cell: the loss never sees building cells, yet in a 0.5 m case a 1 m block that straddles a
    wall still holds open sub-cells that are scored. The fill uses the input only, never the simulation.

    The caller chooses ``model.eval()`` or ``model.train()`` (S14 keeps dropout on for MC dropout).
    """
    if device is None:
        device = next(model.parameters()).device
    preds = []
    for xi, back in _variants(c["x"], c["angle_deg"], mirror_tta):
        raw = predict_tiled(model, xi, device, window, border) if window else predict_whole(model, xi, device, margin)
        preds.append(back(raw))
    pred = np.mean(preds, axis=0).astype(np.float32)
    return fill_buildings(pred, c["x"][BUILDING_CHANNEL] > 0) if fill else pred


def upsample2_periodic(a: np.ndarray) -> np.ndarray:
    """Bilinear 2x upsampling of a periodic field, the same as torch's ``align_corners=False``, with wrap-around.

    Each 0.5 m sub-cell sits a quarter of a 1 m cell away from its parent's centre, so it takes 3/4
    of its parent and 1/4 of the neighbour on its side. Against copying the parent to all four
    sub-cells, this cuts the error a perfect 1 m prediction makes on the 0.5 m grid by more than
    40% (S11 README).
    """
    for axis in (a.ndim - 2, a.ndim - 1):
        before, after = np.roll(a, 1, axis), np.roll(a, -1, axis)
        # stacking right after `axis` and merging the two puts sub-cell 2i before sub-cell 2i+1
        pairs = np.stack([0.75 * a + 0.25 * before, 0.75 * a + 0.25 * after], axis=axis + 1)
        shape = list(a.shape)
        shape[axis] *= 2
        a = pairs.reshape(shape)
    return a


def predict_arrays(model: torch.nn.Module, c: dict[str, Any], device: torch.device | str | None = None,
                   fill: bool = True, mirror_tta: bool = False, upsample: str = "bilinear", **inference) -> np.ndarray:
    """U/u_tau for one whole case, on the grid and in the orientation of the raw files.

    ``c`` holds the arrays of one case with its inputs: ``load_cached(case)`` for a development
    case, or ``with_inputs(build_arrays(load_case(case)))`` for a test case at final scoring, so
    test cases never need to enter the cache. The canonical prediction (``predict_canonical``) is
    turned back out of the canonical frame, and for 0.5 m cases brought to the 0.5 m grid with
    ``upsample``: "bilinear" (periodic bilinear interpolation, the default) or "copy" (each 1 m value
    on its four sub-cells). The result lines up cell by cell with ``load_case(case)["U"]``, which is
    what ``urbanwind.evaluate`` compares against.
    """
    y = from_canonical(predict_canonical(model, c, device, fill, mirror_tta, **inference), c["k"], c["transposed"])
    if np.isclose(c["dx_native"], FINE_DX):
        if upsample not in ("bilinear", "copy"):
            raise ValueError(f"upsample must be 'bilinear' or 'copy', not {upsample!r}")
        y = upsample2_periodic(y) if upsample == "bilinear" else y.repeat(2, axis=0).repeat(2, axis=1)
    if y.shape != tuple(c["native_shape"]):
        raise ValueError(f"prediction shape {y.shape} does not match the native grid {tuple(c['native_shape'])}")
    return np.ascontiguousarray(y)


def predict_native(model: torch.nn.Module, case: str, cache_dir: pathlib.Path = CACHE_DIR,
                   device: torch.device | str | None = None, fill: bool = True, mirror_tta: bool = False,
                   upsample: str = "bilinear", **inference) -> np.ndarray:
    """``predict_arrays`` for a cached development case; ``inference`` takes window, border and margin."""
    return predict_arrays(model, load_cached(case, cache_dir), device, fill, mirror_tta, upsample, **inference)
