"""Tests for the S11 network, loss and trainer (urbanwind.dl.nets, losses and train)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # PyTorch has no wheels for Intel Macs; CI and Apple silicon have it

import torch.nn.functional as F  # noqa: E402

from urbanwind.dl.losses import masked_l1  # noqa: E402
from urbanwind.dl.nets import UNet, build_model, register, upsample_periodic  # noqa: E402
from torch.optim.swa_utils import AveragedModel  # noqa: E402

from urbanwind.dl.train import DEFAULTS, ema_average, fold_table, inner_split, train_epoch  # noqa: E402


def test_unet_keeps_the_grid_and_predicts_positive_speeds():
    model = build_model("unet", 5).eval()
    x = torch.randn(2, 5, 64, 96)
    out = model(x)
    assert out.shape == (2, 1, 64, 96) and (out > 0).all()
    assert 8_000_000 < sum(p.numel() for p in model.parameters()) < 9_000_000
    with pytest.raises(ValueError):
        model(torch.randn(1, 5, 64, 72))  # 72 is not a multiple of 16


def test_unet_is_periodic():
    # shifting a periodic input by a multiple of 16 cells must shift the output by the same amount;
    # zero padding or clamped upsampling anywhere in the network would break this at the edges
    # (in float64, so rounding cannot hide or fake a difference)
    torch.manual_seed(0)
    model = UNet(widths=(8, 16, 16, 32), bottleneck=32).double().eval()
    x = torch.randn(1, 5, 64, 96, dtype=torch.float64)
    with torch.no_grad():
        shifted = model(torch.roll(x, shifts=(16, 48), dims=(2, 3)))
        expected = torch.roll(model(x), shifts=(16, 48), dims=(2, 3))
    assert torch.allclose(shifted, expected, atol=1e-10)


def test_periodic_upsampling_matches_upsampling_a_tiled_field():
    x = torch.randn(1, 3, 8, 12)
    tiled = F.interpolate(x.repeat(1, 1, 3, 3), scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.allclose(upsample_periodic(x), tiled[..., 16:32, 24:48], atol=1e-6)


def test_registry_refuses_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("no_such_net", 5)
    with pytest.raises(ValueError, match="already registered"):
        register("unet")(UNet)


def test_masked_l1_by_hand():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).half()
    y = torch.tensor([[0.0, 2.0], [5.0, 4.0]]).half()  # both in half precision, as under autocast
    mask = torch.tensor([[True, True], [True, False]])  # the last cell is a building
    loss = masked_l1(pred, y, mask)
    assert loss.dtype == torch.float32 and loss.item() == pytest.approx((1 + 0 + 2) / 3)
    assert masked_l1(pred, y, torch.zeros_like(mask)).item() == 0.0  # no valid cell: 0, not NaN
    # the difference itself is taken in float32: in half precision 1000.5 - 2**-10 rounds back to 1000.5
    big, tiny = torch.tensor([1000.5]).half(), torch.tensor([2.0**-10]).half()
    assert masked_l1(big, tiny, torch.tensor([True])).item() == pytest.approx(1000.5 - 2.0**-10, abs=1e-6)


def test_inner_validation_holds_whole_groups_from_the_training_folds():
    folds = fold_table("grouped")
    for fold in range(5):
        train = folds[folds["fold"] != fold]
        fit, val = inner_split(train, 0.15, fold, by_group=True)
        assert set(fit) | set(val) == set(train.index) and not set(fit) & set(val)
        assert not set(folds.loc[fit, "split_group"]) & set(folds.loc[val, "split_group"])
        assert len(val) >= 0.15 * len(train)
        assert inner_split(train, 0.15, fold, by_group=True) == (fit, val)  # the same for every seed


def test_random_split_ignores_groups_and_is_fixed():
    grouped, random = fold_table("grouped"), fold_table("random")
    assert sorted(random["fold"].value_counts()) == [96, 96, 96, 97, 97]
    assert (random["fold"] == fold_table("random")["fold"]).all()
    # a random split puts cases of one city into several folds; the grouped split never does
    assert random.groupby("split_group")["fold"].nunique().max() > 1
    assert grouped.groupby("split_group")["fold"].nunique().max() == 1


def test_a_few_steps_on_one_batch_drive_the_loss_down():
    torch.manual_seed(0)
    model = UNet(widths=(8, 16, 16, 32), bottleneck=32, dropout=0.0)
    x, y = torch.randn(2, 5, 32, 32), torch.rand(2, 1, 32, 32) + 0.5
    mask = torch.ones_like(y, dtype=torch.bool)
    loader = [(x, y, mask)] * 10
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    cfg = DEFAULTS | {"amp": False}
    ema = AveragedModel(model, avg_fn=ema_average(0.999))
    first = train_epoch(model, ema, loader[:1], opt, scaler, masked_l1, torch.device("cpu"), cfg)
    for _ in range(5):
        last = train_epoch(model, ema, loader, opt, scaler, masked_l1, torch.device("cpu"), cfg)
    assert np.isfinite(last) and last < 0.5 * first


def test_a_stopped_fold_resumes_exactly_where_it_left_off(tmp_path, monkeypatch):
    import functools

    from urbanwind.dl import train
    from urbanwind.dl.cache import CACHE_VERSION, _splits_hash, build_arrays, load_cached, write_case
    from urbanwind.dl.datasets import PatchDataset

    rng = np.random.default_rng(0)
    cases = [f"fake{i}" for i in range(3)]
    for case in cases:  # small open fields with a few buildings, written as a cache
        H = np.where(rng.random((64, 64)) < 0.1, 10.0, 0.0)
        U = np.where(H == 0, rng.random((64, 64)) + 0.5, np.nan)
        c = {"H": H, "dx": 1.0, "wind_dir_deg": 0.0, "U": U, "u": U, "v": U * 0, "TKE": U}
        meta = {"u_tau": 0.2, "subset": "idealized", "split_group": "fake", "status": "ok", "fold": 0,
                "version": CACHE_VERSION, "splits_hash": _splits_hash()}
        write_case(case, build_arrays(c) | meta, tmp_path)
    monkeypatch.setattr(train, "PatchDataset", functools.partial(PatchDataset, cache_dir=tmp_path))
    monkeypatch.setattr(train, "load_cached", lambda case: load_cached(case, tmp_path))
    cfg = train.DEFAULTS | {"model": {"name": "unet", "kwargs": {"widths": [8, 16, 16, 32], "bottleneck": 32,
                                                                   "dropout": 0.0}},
                            "amp": False, "max_epochs": 3, "patch": 32, "border": 4, "batch": 4}
    cpu = torch.device("cpu")
    straight, _ = train.fit_fold(cfg, 0, cases[:2], cases[2:], cpu, tmp_path / "a.pt")

    def crash_after_first_epoch(epoch, val):
        if epoch == 1:
            raise KeyboardInterrupt  # stands in for a restart or a closed laptop
    with pytest.raises(KeyboardInterrupt):
        train.fit_fold(cfg, 0, cases[:2], cases[2:], cpu, tmp_path / "b.pt", report=crash_after_first_epoch)
    with pytest.raises(ValueError, match="other settings"):
        train.fit_fold(cfg | {"lr": 5e-4}, 0, cases[:2], cases[2:], cpu, tmp_path / "b.pt")
    resumed, info = train.fit_fold(cfg, 0, cases[:2], cases[2:], cpu, tmp_path / "b.pt")
    assert info["epochs_run"] == 3 and list(resumed["epoch"]) == [0, 1, 2]
    assert np.allclose(resumed["train_loss"], straight["train_loss"], rtol=1e-5)
    assert np.allclose(resumed["val"], straight["val"], rtol=1e-5)


def test_inner_validation_is_balanced_and_no_city_dominates():
    folds = fold_table("grouped")
    for fold in range(5):
        train = folds[folds["fold"] != fold]
        fit, val = inner_split(train, 0.15, fold, by_group=True, cap=0.25)
        held = folds.loc[val]
        for subset, part in train.groupby("subset"):
            share = (held["subset"] == subset).sum() / len(part)
            assert share >= 0.15, (fold, subset, share)
        assert held["split_group"].value_counts().max() <= 0.25 * 0.15 * len(train)
    assert inner_split(folds[folds["fold"] != 0], 0.0, 0, by_group=True)[1] == []  # no validation at all


def test_weight_average_forgets_the_starting_weights():
    average = ema_average(0.999)
    start, now = torch.tensor(1.0), torch.tensor(1.0)
    for n in range(1750):  # one Optuna trial; the training weights sit at 0 while the average starts at 1
        now = average(now, torch.tensor(0.0), torch.tensor(n + 1))
    assert now < 0.01 * start
    assert ema_average(0.999)(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(10**6)) == pytest.approx(0.001)


def test_validation_averages_each_subset_first(monkeypatch):
    from urbanwind.dl import train

    def offset_prediction(model, c, device, fill=False, **kwargs):
        return c["y"] + c["offset"]
    monkeypatch.setattr(train, "predict_canonical", offset_prediction)
    y, valid = np.zeros((4, 4), np.float32), np.ones((4, 4), bool)
    cases = [{"y": y, "valid": valid, "subset": "idealized", "offset": 1.0},
             {"y": y, "valid": valid, "subset": "idealized", "offset": 3.0},
             {"y": y, "valid": valid, "subset": "realistic", "offset": 10.0}]
    cfg = train.DEFAULTS | {"amp": False}
    # per subset: (1 + 3) / 2 = 2 and 10, then (2 + 10) / 2 = 6, not the plain mean 4.67
    assert train.validate(torch.nn.Identity(), cases, torch.device("cpu"), cfg) == pytest.approx(6.0)


def test_tuning_scores_the_last_epochs_and_prunes_cut_trials(monkeypatch):
    import time

    import optuna

    from pipeline.S11_unet_baseline import tune_optuna

    def fake_fit(cfg, seed, fit, val, device, ckpt, report):
        for epoch, value in enumerate([5.0, 4.0, 3.0, 2.0, 1.0]):
            report(epoch, value)
    monkeypatch.setattr(tune_optuna, "fit_fold", fake_fit)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(pruner=optuna.pruners.NopPruner())
    later = tune_optuna.make_objective([], [], torch.device("cpu"), 5, time.time() + 3600, [], smoke=False)
    study.optimize(later, n_trials=1)
    assert study.trials[0].value == pytest.approx(2.0)  # mean of the last three epochs: (3 + 2 + 1) / 3
    passed = tune_optuna.make_objective([], [], torch.device("cpu"), 5, time.time() - 1, [], smoke=False)
    study.optimize(passed, n_trials=1)
    cut = study.trials[1]
    assert cut.state == optuna.trial.TrialState.PRUNED and cut.user_attrs["stopped_by_deadline_after_epochs"] == 1
    assert study.best_trial.number == 0  # a cut trial is never ranked
