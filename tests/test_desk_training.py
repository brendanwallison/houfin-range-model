"""Tests for the DESK training augmentation, split, and loss reparametrization.

Covers the cluster-free pure cores of the training recipe: structured channel-group masking
(``augment.ChannelGroupMasker``), the spatially blocked + buffered validation split
(``augment.blocked_holdout``), the 5x5 partial conv, and the stabilizing-loss
reparametrization. Runs standalone or under pytest; no GPU and no state files needed.
"""
import json
import os
import re
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.augment import (ChannelGroupMasker, blocked_holdout)
from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder, PartialConv2d

TEMPS = ("Tave", "Tmax", "Tmin")
OTHERS = ("CMD", "CMI", "DD18", "DD5", "DDsub0", "DDsub18", "Eref", "NFFD", "PAS", "PPT", "RH")



def _metric_dict(x, m, years):
    """The (years, flat cell idx, rows) pool train_model_ema takes: every masked cell in
    every year, which is what the dense single-year grid used to mean implicitly."""
    H, W = m.shape
    flat = np.flatnonzero(m.reshape(-1))
    rows = x.reshape(H * W, -1)[flat].astype("float32")
    ys = np.repeat(np.asarray(years, dtype=np.int64), len(flat))
    return (ys, np.tile(flat, len(years)), np.tile(rows, (len(years), 1)))


def _schema(bio_start=8):
    """The pre-BUI layout: 14 climate bases (3 with q10/q90) = 240 ch, + 55 others = 295.

    Deliberately without the ``bui`` stream: these tests check group CONSTRUCTION from a
    schema, so the fixture only needs to be a realistic one. ``_schema_with_indicator``
    adds the BUI-shaped stream for the tests that need an availability channel.
    """
    months = [(k, (bio_start - 1 + k - 1) % 12 + 1) for k in range(1, 13)]
    cvars = []
    for b in sorted(TEMPS + OTHERS):
        for lvl in (("q10", "q50", "q90") if b in TEMPS else ("q50",)):
            cvars += [f"{b}_b{k:02d}m{m:02d}_{lvl}" for k, m in months]
    cvars = sorted(cvars)                                  # discover_variables order
    spec = [("climate", cvars),
            ("landuse", [f"lu{i}" for i in range(33)]),
            ("hyde", ["population_density", "rural_population", "urban_population"]),
            ("soil", []), ("elevation", [])]
    dims = {"soil": 16, "elevation": 3}
    streams, start = [], 0
    for name, v in spec:
        d = dims.get(name, len(v))
        streams.append({"name": name, "start": start, "end": start + d, "dim": d,
                        "variables": v})
        start += d
    return {"streams": streams, "total_dim": start}


def test_masker_groups():
    sch = _schema()
    M = ChannelGroupMasker(sch, {"enabled": True})
    assert M.C == 295, M.C
    assert len(M.base_keys) == 14 and len(M.month_keys) == 12
    # q50 is deliberately NOT maskable as a level: every base has it, so dropping it would
    # remove most of climate in one draw.
    assert M.level_keys == ["q10", "q90"]
    # climate is 240/295 channels -- dropping it as a stream would blind the model
    assert "climate" not in M.stream_keys and len(M.stream_keys) == 4

    # a temperature base spans 12 months x 3 levels; a flux base 12 months x 1 level
    assert M.by_base["Tmax"].sum() == 36
    assert M.by_base["PPT"].sum() == 12
    # one month position touches every base at every level: 11*1 + 3*3
    assert M.by_month_pos[6].sum() == 20
    assert M.by_level["q10"].sum() == 36                   # 3 temps x 12 months

    # b01 must be the bio-year START (August with bio_year_start_month=8), not January
    names = sch["streams"][0]["variables"]
    b01 = [n for n in names if "_b01m" in n]
    assert all("m08" in n for n in b01), b01[:3]
    print("masker group construction OK")


def test_masker_draw_properties():
    M = ChannelGroupMasker(_schema(), {"enabled": True, "max_masked_frac": 0.5})
    rng = np.random.default_rng(0)
    fracs = [M.sample_drop(rng).mean() for _ in range(2000)]
    assert max(fracs) <= 0.5 + 1e-9, max(fracs)            # cap honoured
    assert 0.05 < np.mean(fracs) < 0.45, np.mean(fracs)    # actually masking something

    climate = M.by_stream["climate"]
    for _ in range(2000):
        assert not M.sample_drop(rng)[climate].all()       # never fully blinded

    # keep is strictly 0/1: survivors are NOT rescaled (denoising augmentation, not dropout)
    keep = M.sample_keep(rng)
    assert keep.shape == (1, 1, 1, 295)
    assert set(torch.unique(keep).tolist()) <= {0.0, 1.0}

    # must not mutate the caller's tensor -- in the trainer the input is a VIEW of the single
    # resident year window, so an in-place mask would destroy that year for all later epochs
    x = torch.randn(1, 3, 4, 295)
    x0 = x.clone()
    y = x * M.sample_keep(rng)
    assert torch.equal(x, x0) and y.shape == x.shape

    # disabled -> exact identity
    off = ChannelGroupMasker(_schema(), {"enabled": False})
    assert off.sample_drop(rng).sum() == 0
    assert torch.equal(x * off.sample_keep(rng), x)
    print("masker draw properties OK")


def test_masker_span_is_contiguous():
    """A span mask must hit consecutive bio-year positions, with no wraparound."""
    M = ChannelGroupMasker(_schema(), {
        "enabled": True, "p_base": 0.0, "p_month": 0.0, "p_level": 0.0, "p_stream": 0.0,
        "span_prob": 1.0, "span_max": 3})
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(300):
        drop = M.sample_drop(rng)
        hit = sorted(p for p in M.month_keys if (drop & M.by_month_pos[p]).any())
        assert hit, "span draw masked nothing"
        assert hit == list(range(hit[0], hit[0] + len(hit))), hit   # contiguous
        assert 1 <= len(hit) <= 3
        seen.add((hit[0], len(hit)))
    # b12 and b01 are 11 months apart in the window, so they must never co-occur alone
    assert not any(h == (12, 2) for h in seen)
    print("masker contiguous span OK")


def _schema_with_indicator(n_val=6):
    """A schema plus a BUI-shaped stream: ``n_val`` value channels + one availability
    channel, declared as the stream's ``indicator_variable``."""
    sch = _schema()
    variables = [f"bui_q{q:02d}" for q in (5, 25, 50, 75, 90, 99)][:n_val] + ["bui_avail"]
    start = sch["total_dim"]
    sch["streams"].append({"name": "bui", "start": start, "end": start + len(variables),
                           "dim": len(variables), "variables": variables,
                           "indicator_variable": "bui_avail"})
    sch["total_dim"] = start + len(variables)
    return sch


def test_persistent_tier_holds_across_years_and_transient_does_not():
    """Persistence is the axis today's augmentation cannot express: a per-year draw can
    only ever present a covariate as transiently absent, while BUI is missing in EVERY
    year over the same region."""
    sch = _schema()
    M = ChannelGroupMasker(sch, {
        "enabled": True, "p_base": 0.0, "p_month": 0.0, "span_prob": 0.0, "p_level": 0.0,
        "p_stream": 0.5,
        "persist": {"p_stream": 0.5}})
    assert M.persist_enabled
    rng = np.random.default_rng(0)

    # one "optimizer step": a single persistent draw reused for every year
    step = M.sample_persistent_keep(rng)
    years = [M.sample_keep(rng) for _ in range(12)]
    assert not all(torch.equal(years[0], y) for y in years[1:]), "transient tier never varied"
    # whatever the persistent draw dropped stays dropped in every year of the window
    dropped = (step[0, 0, 0] == 0.0)
    assert bool(dropped.any()), "persistent draw masked nothing at p_stream=0.5"
    for y in years:
        assert float(((y * step)[0, 0, 0])[dropped].abs().max()) == 0.0

    # a second step draws a different persistent mask
    steps = [M.sample_persistent_keep(rng) for _ in range(12)]
    assert not all(torch.equal(steps[0], s) for s in steps[1:]), "persistent tier never varied"

    # the composition the trainer performs: kept only if BOTH tiers keep it
    combined = years[0] * step
    assert set(torch.unique(combined).tolist()) <= {0.0, 1.0}
    assert bool((combined <= years[0]).all()) and bool((combined <= step).all())


def test_persist_absent_is_exactly_the_old_behaviour():
    """No ``persist`` block -> no persistent mask at all, so the default config path is
    bit-for-bit the pre-persistence one."""
    M = ChannelGroupMasker(_schema(), {"enabled": True})
    assert M.persist_enabled is False
    assert M.sample_persistent_keep(np.random.default_rng(0)) is None
    # an all-zero persist block is also a no-op, not a mask of everything
    Z = ChannelGroupMasker(_schema(), {"enabled": True, "persist": {"p_stream": 0.0}})
    assert Z.persist_enabled is False and Z.sample_persistent_keep(np.random.default_rng(0)) is None
    # ...and disabled overall stays identity even with persistence configured
    off = ChannelGroupMasker(_schema(), {"enabled": False, "persist": {"p_stream": 1.0}})
    assert off.sample_persistent_keep(np.random.default_rng(0)) is None


def test_persist_tier_has_its_own_granularity():
    """Structural absence is regional, so the persistent tier takes its own tile_cells."""
    M = ChannelGroupMasker(_schema(), {
        "enabled": True, "tile_cells": 6, "persist": {"p_stream": 0.5, "tile_cells": 24}})
    assert M.transient.tile_cells == 6 and M.persist.tile_cells == 24
    rng = np.random.default_rng(0)
    tr = M.sample_keep(rng, hw=(48, 48))
    pe = M.sample_persistent_keep(rng, hw=(48, 48))
    assert tr.shape == (1, 48, 48, M.C) and pe.shape == (1, 48, 48, M.C)
    # the coarser tier must be constant over any 24x24 block it owns
    blk = pe[0, :24, :24, :]
    assert torch.equal(blk, blk[0, 0].expand_as(blk))


def test_indicator_is_masked_with_its_values():
    """The indicator and its value channels are one atomic unit. Values gone with the
    indicator still at 1 teaches that 'available' can coexist with a meanless value --
    the single inference the indicator exists to prevent."""
    sch = _schema_with_indicator()
    M = ChannelGroupMasker(sch, {"enabled": True, "p_stream": 0.5,
                                 "persist": {"p_stream": 0.5}})
    (ind_ch, val_mask), = M.indicator_groups
    bui = sch["streams"][-1]
    assert ind_ch == bui["end"] - 1                      # bui_avail is the last channel
    assert val_mask.sum() == bui["dim"] - 1

    rng = np.random.default_rng(0)
    saw_dropped = saw_kept = False
    for _ in range(4000):
        for tier in (M.transient, M.persist):
            drop = M.sample_drop(rng, tier=tier)
            if drop[val_mask].any() or drop[ind_ch]:
                assert drop[ind_ch] and drop[val_mask].all(), "indicator/value split apart"
                saw_dropped = True
            else:
                saw_kept = True
    assert saw_dropped and saw_kept, "test never exercised both states"


def test_indicator_must_name_a_real_channel():
    sch = _schema_with_indicator()
    sch["streams"][-1]["indicator_variable"] = "not_a_channel"
    try:
        ChannelGroupMasker(sch, {"enabled": True})
    except ValueError as exc:
        assert "not_a_channel" in str(exc)
    else:
        raise AssertionError("a bogus indicator_variable was accepted")


def test_streams_without_an_indicator_are_untouched():
    M = ChannelGroupMasker(_schema(), {"enabled": True})
    assert M.indicator_groups == []
    rng = np.random.default_rng(0)
    assert M.sample_drop(rng).shape == (M.C,)


def test_blocked_buffered_split():
    H, W = 133, 224
    valid = np.zeros((H, W), bool)
    valid[5:128, 10:214] = True

    def min_contact_radius(val, train, rmax):
        """Smallest r for which a train cell sits within Chebyshev distance r of a val cell."""
        for r in range(rmax + 1):
            dil = np.zeros_like(val)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    sl = np.zeros_like(val)
                    ys, ys2 = slice(max(0, dy), min(H, H + dy)), slice(max(0, -dy), min(H, H - dy))
                    xs, xs2 = slice(max(0, dx), min(W, W + dx)), slice(max(0, -dx), min(W, W - dx))
                    sl[ys, xs] = val[ys2, xs2]
                    dil |= sl
            if (dil & train).any():
                return r
        return None

    # The buffer must track the kernel: with kernel 5 the receptive radius is 2, so no train cell
    # may lie within 2 of a val cell. Shrinking the kernel must shrink the buffer, not keep it.
    for kernel, want_buf in ((5, 2), (3, 1), (0, 0)):
        buf = kernel // 2
        assert buf == want_buf
        val, bufm = blocked_holdout(valid, block_cells=12, holdout_frac=0.2,
                                    buffer_cells=buf, seed=0)
        train = valid & ~val & ~bufm
        assert not (val & bufm).any() and not (val & train).any() and not (bufm & train).any()
        assert val.any() and train.any()
        r = min_contact_radius(val, train, buf + 1)
        assert r is None or r > buf, f"kernel={kernel}: train cell within {r} of val (buffer {buf})"
        if buf == 0:
            assert bufm.sum() == 0
        else:
            assert bufm.sum() > 0

    # block structure: an i.i.d. draw produces isolated cells, a blocked one does not
    val, _ = blocked_holdout(valid, 12, 0.2, 2, seed=0)
    nb = np.zeros_like(val, int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nb[1:-1, 1:-1] += val[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx].astype(int)
    assert not (val[1:-1, 1:-1] & (nb[1:-1, 1:-1] == 1)).any()

    # deterministic in the seed, and the seed actually matters
    a = blocked_holdout(valid, 12, 0.2, 2, seed=0)
    assert np.array_equal(a[0], blocked_holdout(valid, 12, 0.2, 2, seed=0)[0])
    assert not np.array_equal(a[0], blocked_holdout(valid, 12, 0.2, 2, seed=1)[0])
    # holdout never escapes the valid footprint
    assert not (a[0] & ~valid).any() and not (a[1] & ~valid).any()
    print("blocked + buffered split OK")


def test_partial_conv_5x5():
    c = PartialConv2d(4, 4, 5, padding=2).eval()
    assert c._winsize == 25.0 and c.kernel_size == (5, 5) and c.padding == (2, 2)
    x = torch.randn(1, 4, 11, 11)
    full = torch.ones(1, 1, 11, 11)
    with torch.no_grad():
        out, newm = c(x, full)
        ref = torch.nn.functional.conv2d(x, c.weight, c.bias, padding=2)
    assert out.shape == x.shape and newm.shape == full.shape
    # a fully-valid window needs no renormalization: interior must equal a plain conv
    assert torch.allclose(out[..., 2:-2, 2:-2], ref[..., 2:-2, 2:-2], atol=1e-5)
    # a fully-invalid mask propagates invalidity rather than inventing values
    with torch.no_grad():
        out0, m0 = c(x, torch.zeros_like(full))
    assert float(out0.abs().max()) == 0.0 and float(m0.max()) == 0.0
    print("5x5 partial conv OK")


def test_dropout_threading_and_loss_reparam():
    m = MultiStreamAutoencoder([240, 33, 3, 16, 3], 64, spatial_kernel=5, dropout=0.1)
    rates = {b.drop.p for enc in m.encoders for b in enc if hasattr(b, "drop")}
    assert rates == {0.1} and m.dropout == 0.1
    assert MultiStreamAutoencoder([4], 8).dropout == 0.5        # legacy default preserved

    # The reparametrization must be exactly neutral: dividing the stabilizing term by latent_dim
    # while multiplying its weight by latent_dim leaves the total loss unchanged.
    latent = 64
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(500, latent, generator=g)
    tgt = torch.randn(500, latent, generator=g)
    per_cell_sum = torch.sum((pred - tgt) ** 2, dim=1).mean()
    old = 1.0 * per_cell_sum
    new = float(latent) * (per_cell_sum / latent)
    assert torch.allclose(old, new, rtol=0, atol=1e-6), (float(old), float(new))
    print("dropout threading + loss reparametrization OK")


def _tiny_train(augment, seed=0, epochs=3, patch=None, holdout_year_targets=None,
                convert_targets=True):
    """Drive train_model_ema on a small synthetic window; return the per-epoch Stab values."""
    import contextlib
    import io
    import re

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 4, 16, 20, 64
    rng = np.random.default_rng(0)                       # data fixed across runs
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    msk = np.ones((T, H, W), bool)
    years = list(range(2022, 2026))
    m = np.ones((H, W), bool)
    w = np.ones((H, W), dtype="float32")
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32") * 0.1, m, m, w)
           for y in years[1:]}
    x = rng.random((H, W, 12)).astype("float32")

    # the loop calls spacetime_kernel_loss now -- patching true_kernel_loss would
    # patch a function nothing on the hot path calls, and the guard would never fire
    saved = D.spacetime_kernel_loss
    if patch is not None:
        D.spacetime_kernel_loss = patch
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            D.train_model_ema(
                cov, msk, years, tgt, _metric_dict(x, m, years), m, m, dims, latent_dim=L,
                ema_cfg={"earlystop_warmup": 1}, spatial_kernel=5, epochs=epochs, lr=1e-3,
                seed=seed, weights={"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1},
                schema=sch, augment_cfg={"enabled": augment}, dropout=0.1,
                warmup_epochs=1, min_lr_frac=0.05,
                holdout_year_targets=holdout_year_targets,
                _skip_target_conversion=not convert_targets)
    finally:
        D.spacetime_kernel_loss = saved
    text = out.getvalue()
    return [float(v) for v in re.findall(r"Stab (\d+\.\d+)", text)], text


def test_train_loop_and_ablation():
    """The trainer runs, is seed-reproducible, and augment=false recovers the old recipe.

    The ablation guard is what keeps the monthly-vs-annual comparison recoverable: this whole
    change set landed in one run, so being able to reproduce the un-augmented recipe exactly is
    the only way to attribute a difference to the covariates rather than to the recipe.
    """
    off1, text = _tiny_train(False)
    off2, _ = _tiny_train(False)
    on1, _ = _tiny_train(True)
    assert len(off1) == 3
    assert off1 == off2, (off1, off2)                    # bit-reproducible for a fixed seed
    assert on1 != off1                                   # masking actually perturbs training
    assert _tiny_train(True, seed=7)[0] == _tiny_train(True, seed=7)[0]   # ON is seeded too

    # the per-epoch line must carry the diagnostics that were previously missing entirely
    for token in ("Val(all-yr)", "Val(yr-out)", "mix ", "lr ", "RSS "):
        assert token in text, token
    # LR must warm up then decay (warmup_epochs=1 -> epoch 1 at full lr, then cosine)
    lrs = [float(v) for v in __import__("re").findall(r"lr ([\d.e-]+)", text)]
    assert lrs[0] > lrs[-1], lrs
    print("train loop + ablation guard OK")


def test_rotation_diagnostic_reads_a_known_angle():
    """`rotT` must equal the target's true rotation, and `rotP` must start near zero.

    Built against ground truth: targets are constructed by rotating a fixed unit vector by a
    known angle between the deep and anchor year, so `1 - cos` is known exactly. This is the
    instrument that was missing -- a per-cell MSE cannot expose a temporal-amplitude deficit,
    because Z is dominated by the static spatial pattern and MSE's optimum for a
    poorly-predicted component is a shrunk one.
    """
    import contextlib
    import io
    import re

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 6, 20, 26, 64
    rng = np.random.default_rng(0)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    msk = np.ones((T, H, W), bool)
    years = list(range(2020, 2026))
    m = np.ones((H, W), bool)

    ang = np.deg2rad(50.0)
    base = rng.normal(size=(H, W, L)).astype("float32")
    base /= np.linalg.norm(base, axis=-1, keepdims=True)
    perp = rng.normal(size=(H, W, L)).astype("float32")
    perp -= (perp * base).sum(-1, keepdims=True) * base
    perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
    tgt = {}
    for i, y in enumerate(years[1:]):
        a = ang * (i / max(len(years) - 2, 1))
        tgt[y] = ((np.cos(a) * base + np.sin(a) * perp).astype("float32"), m, m,
                  np.ones((H, W), dtype="float32"))

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        D.train_model_ema(
            cov, msk, years, tgt,
            _metric_dict(rng.random((H, W, 12)).astype("float32"), m, years), m, m, dims,
            latent_dim=L, ema_cfg={"earlystop_warmup": 1}, spatial_kernel=5, epochs=3, lr=1e-3,
            weights={"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1},
            schema=sch, augment_cfg={"enabled": True}, dropout=0.1,
            warmup_epochs=1, min_lr_frac=0.05)
    text = out.getvalue()

    assert "rotation/direction diagnostic 2021->2025" in text
    # `rot tr R va R ho Rh (raw Rr/Rrv)`: R is z_ema (the SUPERVISED quantity, so 1.0 is the
    # goal), raw is the pre-EMA rotation, which must be LARGER because the EMA attenuates. `ho`
    # is the withheld-era pair and is nan here (no temporal holdout in this fixture).
    rots = re.findall(
        r"rot tr ([\d.]+) va ([\d.]+) ho (?:[\d.]+|nan) \(raw ([\d.]+)/([\d.]+)\)", text)
    assert len(rots) == 3, text
    ema_r = [float(a) for a, _, _, _ in rots]
    raw_r = [float(c) for _, _, c, _ in rots]
    # an untrained model barely rotates, and the ratio must grow as it learns
    assert ema_r[0] < 0.1, ema_r
    assert raw_r[-1] > raw_r[0], raw_r
    # the EMA can only attenuate, so the smoothed ratio never exceeds the raw one
    assert all(e <= r + 1e-9 for e, r in zip(ema_r, raw_r)), (ema_r, raw_r)
    # the target rotation is a fixed property of the data; the diagnostic line reports the
    # cell counts, and the mse columns must be finite
    assert re.search(r"mse tr [\d.]+ va\(pool\) [\d.]+ gap [\d.]+x", text), text
    # The pooled val MSE must be split into its trained-era (E) and withheld-era (H) parts --
    # reporting only the pool is what forced the sweep's two key numbers to be recovered by hand.
    assert len(re.findall(r"va\(sp\) [\d.]+ va\(sp\+t\) (?:[\d.]+|nan)", text)) == 3, text
    # the direction column and its permuted null must both be present every epoch
    assert len(re.findall(
        r"dcos tr [\d.-]+ va [\d.-]+ ho (?:[\d.-]+|nan) null [+-][\d.]+/[+-][\d.]+",
        text)) == 3, text
    # the MSE-calibration ratio rot/dcos^2 is reported (diagnostic only, never optimized)
    assert len(re.findall(r"cal (?:[\d.]+|nan|inf)", text)) == 3, text
    # and the MAGNITUDE half of the same pairs: rot/dcos/cal are all angular, so without this
    # the line accounts for only one of the two terms of the error and cannot separate "moved the
    # wrong way" from "barely moved".
    assert len(re.findall(r"mag tr (?:[\d.]+|nan) va (?:[\d.]+|nan)", text)) == 3, text
    print("rotation diagnostic reads a known angle OK")


def test_capacity_knob_and_width_persistence():
    """hidden_width is the capacity lever, and it MUST be persisted to reload weights."""
    dims = [240, 33, 3, 16, 3]
    wide = MultiStreamAutoencoder(dims, 64, spatial_kernel=5, dropout=0.1)
    narrow = MultiStreamAutoencoder(dims, 64, spatial_kernel=5, dropout=0.1, hidden_width=128)
    assert wide.hidden_width == 256, "default must stay max(128, latent_dim*4)"
    assert narrow.hidden_width == 128

    n_wide = sum(p.numel() for p in wide.parameters())
    n_narrow = sum(p.numel() for p in narrow.parameters())
    assert n_narrow < n_wide / 3, (n_wide, n_narrow)      # branch params scale as h**2

    # Rebuilding at the DEFAULT width cannot load narrow weights -- this is exactly why
    # hidden_width/mlp_expansion go into desk_meta.npz and are read back at inference.
    sd = narrow.state_dict()
    try:
        MultiStreamAutoencoder(dims, 64, spatial_kernel=5).load_state_dict(sd)
        raise AssertionError("default-width rebuild must reject h=128 weights")
    except RuntimeError:
        pass
    MultiStreamAutoencoder(dims, 64, spatial_kernel=5, hidden_width=128).load_state_dict(sd)

    # mlp_expansion is the second width knob and must also round-trip
    thin = MultiStreamAutoencoder(dims, 64, spatial_kernel=5, hidden_width=128, mlp_expansion=2)
    assert thin.mlp_expansion == 2
    assert sum(p.numel() for p in thin.parameters()) < n_narrow

    # still functional at the reduced width
    x = torch.randn(1, 8, 9, sum(dims)); m = torch.ones(1, 8, 9, dtype=torch.bool)
    narrow.eval()
    with torch.no_grad():
        z, rec = narrow(x, m)
    assert z.shape == (1, 8, 9, 64) and rec.shape == x.shape
    print("capacity knob + width persistence OK")


def test_direction_cosine_recovers_known_angles():
    """`dcos` must measure DIRECTION and ignore magnitude.

    `rot` alone is a magnitude ratio: two models can match it while moving in opposite
    directions, which is why an untrained model scores ~0.9 on it (a random projection inherits
    the covariates' own rotation). The pair is what is interpretable -- for an MSE objective the
    optimal magnitude of a prediction whose direction cosine is rho is exactly rho, and since
    1-cos ~ theta^2/2 the calibrated relation is rot ~ dcos^2.
    """
    from src.community_encoder.train_DESK.desk_training import median_dir_cos

    g = torch.Generator().manual_seed(0)
    n, L = 4000, 64
    dt = torch.randn(n, L, generator=g); dt /= dt.norm(dim=1, keepdim=True)
    perp = torch.randn(n, L, generator=g)
    perp -= (perp * dt).sum(1, keepdim=True) * dt
    perp /= perp.norm(dim=1, keepdim=True)

    for deg in (0, 30, 60, 90, 120, 180):
        a = np.deg2rad(deg)
        # random per-row magnitudes: a DIRECTION metric must be blind to them
        dp = (np.cos(a) * dt + np.sin(a) * perp) * torch.rand(n, 1, generator=g) * 5
        assert abs(median_dir_cos(dp, dt) - np.cos(a)) < 1e-3, deg

    assert abs(median_dir_cos(dt * 100.0, dt) - 1.0) < 1e-5      # scale-invariant
    # degenerate rows must be skipped, not silently scored as 0
    mixed = torch.cat([dt[:10] * 0.0, dt[10:]])
    assert abs(median_dir_cos(mixed, dt) - 1.0) < 1e-5
    assert np.isnan(median_dir_cos(dt * 0.0, dt))                 # all-degenerate -> NaN
    print("direction cosine recovers known angles OK")


def test_nonfinite_loss_guard():
    """A non-finite loss must skip the step and abort only if it persists."""
    saved = None
    try:
        _tiny_train(True, epochs=20, patch=lambda *a, **k: torch.tensor(float("inf")))
        raise AssertionError("expected RuntimeError after repeated non-finite losses")
    except RuntimeError as exc:
        assert "non-finite" in str(exc), exc
    del saved
    print("non-finite loss guard OK")


if __name__ == "__main__":
    test_masker_groups()
    test_masker_draw_properties()
    test_masker_span_is_contiguous()
    test_persistent_tier_holds_across_years_and_transient_does_not()
    test_persist_absent_is_exactly_the_old_behaviour()
    test_persist_tier_has_its_own_granularity()
    test_indicator_is_masked_with_its_values()
    test_indicator_must_name_a_real_channel()
    test_streams_without_an_indicator_are_untouched()
    test_blocked_buffered_split()
    test_partial_conv_5x5()
    test_dropout_threading_and_loss_reparam()
    test_train_loop_and_ablation()
    test_rotation_diagnostic_reads_a_known_angle()
    test_capacity_knob_and_width_persistence()
    test_spatial_interp_baseline()
    test_direction_cosine_recovers_known_angles()
    test_nonfinite_loss_guard()
    print("\nALL DESK-TRAINING CHECKS PASSED")


# ----------------------------- per-point weights and the split domain -----------------------

def test_point_set_loader_defaults_missing_files_to_neutral(tmp_path):
    """A trend-product point set has no weights or supervise file. It must still load, with
    everything counting and everything supervising -- that is what keeps the A/B possible."""
    from src.community_encoder.train_DESK import desk_training as D
    np.save(tmp_path / "X_points.npy", np.zeros((4, 3), "float32"))
    np.save(tmp_path / "point_index.npy", np.zeros((4, 3), "int32"))
    X, pidx, w, sup = D.load_point_set(str(tmp_path))
    assert w.shape == (4,) and (w == 1.0).all()
    assert sup.shape == (4,) and sup.all()


def test_point_set_loader_refuses_a_length_mismatch(tmp_path):
    """These are parallel arrays in one row order. A short weights file would attach the wrong
    weight to the wrong cell-year with no symptom anywhere downstream."""
    from src.community_encoder.train_DESK import desk_training as D
    np.save(tmp_path / "X_points.npy", np.zeros((4, 3), "float32"))
    np.save(tmp_path / "point_index.npy", np.zeros((4, 3), "int32"))
    np.save(tmp_path / "point_weights.npy", np.ones(3, "float32"))
    try:
        D.load_point_set(str(tmp_path))
    except SystemExit as exc:
        assert "point_weights" in str(exc)
    else:
        raise AssertionError("a short weights file was accepted")


def test_supervised_cells_ignores_unsupervised_rows():
    """Duplicate cell-years are kept for the kernel but only one supervises. The split domain
    must be built from the supervised rows, or it would include cells whose only rows are
    duplicates already covered elsewhere."""
    from src.community_encoder.train_DESK import desk_training as D
    pidx = np.array([[1, 1, 2000], [5, 5, 2000], [9, 9, 2000]], dtype="int32")
    sup = np.array([True, False, True])
    grid = D.supervised_cells(pidx, sup, (10, 10))
    assert grid[1, 1] and grid[9, 9]
    assert not grid[5, 5]
    assert grid.sum() == 2


def test_weights_actually_change_the_stabilizing_loss():
    """Guards against inert plumbing: if the weight grid were threaded through but never
    multiplied in, every test above would still pass. Downweighting the cells that carry the
    error has to lower the loss."""
    import torch
    L, H, W = 4, 6, 6
    g = torch.Generator().manual_seed(0)
    zg = torch.zeros(H, W, L)
    pred = torch.zeros(H, W, L)
    pred[0, 0] = 1.0                       # all the error sits in one cell
    tr = torch.zeros(H, W, dtype=torch.bool); tr[0, 0] = True; tr[1, 1] = True

    def stab(wg):
        s = torch.sum((pred[tr] - zg[tr]) ** 2, dim=1) * wg[tr]
        return float(s.sum() / max(float(wg[tr].sum()), 1e-8) / L)

    full = torch.ones(H, W)
    half = torch.ones(H, W); half[0, 0] = 0.5      # downweight the erroring cell
    assert stab(half) < stab(full), "the weight grid is not reaching the loss"
    # and a uniform rescale must NOT change it -- the denominator is the weight sum, so the
    # effective learning rate on this term stays put when weights are rescaled
    assert abs(stab(full * 0.3) - stab(full)) < 1e-9



# --------- the metric loss is spatiotemporal, not spatial-per-year ---------

def test_the_metric_pool_spans_years_so_pairs_can_cross_time():
    """The ESK basis is ONE joint kernel-PCA over every (cell, year) point, so its contract is
    dot(z_i,z_j) ~= Ruzicka(x_i,x_j) for ANY two points -- two cells in one year, one cell in
    two years, or two cells in two different years. A pool grouped by year could only ever
    enforce the spatial half of that."""
    from src.community_encoder.train_DESK.desk_training import spacetime_metric_pool
    W = 10
    pip = np.array([[0, 0, 1975], [0, 1, 1975], [2, 3, 2025]], dtype=np.int32)
    Xp = np.arange(9, dtype="float32").reshape(3, 3)
    m_tr = np.zeros((5, W), bool); m_tr[0, 0] = m_tr[0, 1] = m_tr[2, 3] = True
    yrs, flat, rows = spacetime_metric_pool(pip, Xp, np.ones(3, bool), m_tr, W)
    assert list(yrs) == [1975, 1975, 2025]           # one flat pool, not a per-year grouping
    assert list(flat) == [0, 1, 2 * W + 3]
    assert rows.shape == (3, 3)


def test_a_cell_surveyed_in_one_year_only_still_enters_the_pool():
    """Under the old single-year anchor this cell was invisible: no row in the anchor year
    meant no contribution to the similarity target at all."""
    from src.community_encoder.train_DESK.desk_training import spacetime_metric_pool
    W = 4
    yrs, flat, _ = spacetime_metric_pool(
        np.array([[0, 0, 1975]], dtype=np.int32), np.ones((1, 2), "float32"),
        np.ones(1, bool), np.ones((1, W), bool), W)
    assert list(yrs) == [1975] and list(flat) == [0]


def test_the_pool_excludes_heldout_and_unsupervised_rows():
    """Held-out cells must not enter the similarity target, or the spatial split stops
    measuring generalization. Duplicate rows kept only for the ESK basis carry supervise=False
    and would otherwise enter the pair pool twice."""
    from src.community_encoder.train_DESK.desk_training import spacetime_metric_pool
    W = 4
    pip = np.array([[0, 0, 2000], [0, 1, 2000], [0, 1, 2000], [0, 2, 2000]], dtype=np.int32)
    m_tr = np.zeros((1, W), bool); m_tr[0, 0] = m_tr[0, 1] = True     # (0,2) held out
    _, flat, _ = spacetime_metric_pool(pip, np.ones((4, 2), "float32"),
                                       np.array([True, True, False, True]), m_tr, W)
    assert list(flat) == [0, 1]


def test_the_kernel_loss_reads_z_from_each_point_own_year():
    """The failure this guards: indexing every point into one year's Z slice. Z is built so
    year 0 matches the communities exactly and year 1 has them swapped, so reading the wrong
    year cannot score zero by luck."""
    import torch
    from src.community_encoder.train_DESK.desk_training import spacetime_kernel_loss
    T, H, W, L = 2, 1, 2, 2
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])           # two disjoint communities -> sim 0
    z = torch.zeros(T, H, W, L)
    z[0, 0, 0] = torch.tensor([1.0, 0.0]); z[0, 0, 1] = torch.tensor([0.0, 1.0])
    z[1, 0, 0] = torch.tensor([1.0, 0.0]); z[1, 0, 1] = torch.tensor([1.0, 0.0])
    flat = torch.tensor([0, 1])
    right = spacetime_kernel_loss(z, torch.tensor([0, 0]), flat, x, 512,
                                  torch.Generator().manual_seed(0))
    wrong = spacetime_kernel_loss(z, torch.tensor([1, 1]), flat, x, 512,
                                  torch.Generator().manual_seed(0))
    assert right.item() < 1e-6, right.item()             # orthogonal z -> dot 0 -> matches
    assert wrong.item() > 0.1, wrong.item()              # identical z -> dot 1 -> way off


def test_pairs_actually_span_years_in_the_sampled_loss():
    """Not just the pool: the SAMPLER must produce cross-year pairs. Two years hold the same
    cell with the same community, so a within-year sampler would score 0 -- only cross-year
    pairs, where z differs between the years, can make this loss positive."""
    import torch
    from src.community_encoder.train_DESK.desk_training import spacetime_kernel_loss
    T, H, W, L = 2, 1, 1, 2
    x = torch.tensor([[1.0, 1.0]])                       # one cell, identical both years
    z = torch.zeros(T, H, W, L)
    z[0, 0, 0] = torch.tensor([1.0, 0.0])                # self-similarity 1 -> dot 1: exact
    z[1, 0, 0] = torch.tensor([0.0, 0.0])                # but across years the dot is 0
    pool_x = torch.cat([x, x])
    loss = spacetime_kernel_loss(z, torch.tensor([0, 1]), torch.tensor([0, 0]), pool_x, 512,
                                 torch.Generator().manual_seed(0))
    assert loss.item() > 0.2, loss.item()


def test_supervision_is_not_gated_on_the_anchor_year_esk_mask():
    """The 2,222-cell ceiling on TACC. compute_valid_mask used to intersect with the ESK Z
    mask, which marks cells holding an ANCHOR-YEAR embedding -- but DESK never reads those
    values, since its per-year targets are projected per point. Gating on it discarded every
    cell BBS surveyed in some other year."""
    from src.community_encoder.train_DESK.desk_training import compute_valid_mask
    H, W, S, C = 4, 4, 3, 2
    target = np.full((H, W, S), np.nan, "float32")
    target[0, 0] = 1.0; target[1, 1] = 1.0; target[2, 2] = 1.0     # observed in some year
    cov = np.zeros((H, W, C), "float32")
    m = compute_valid_mask(target, cov)
    assert int(m.sum()) == 3
    assert m[0, 0] and m[1, 1] and m[2, 2]


def test_a_cell_without_covariates_is_still_excluded():
    """The covariate intersection is real: the model cannot predict a cell it has no inputs
    for, so such a cell must not enter supervision or the validation set."""
    from src.community_encoder.train_DESK.desk_training import compute_valid_mask
    target = np.ones((2, 2, 3), "float32")
    cov = np.zeros((2, 2, 2), "float32"); cov[1, 1, 0] = np.nan
    m = compute_valid_mask(target, cov)
    assert int(m.sum()) == 3 and not m[1, 1]


def test_temporally_withheld_years_never_reach_the_metric_pool():
    """The leak that would have invalidated the whole temporal-extrapolation experiment.
    Zeroing a year's train mask in `targets` keeps it out of the stabilizing loss, but the
    metric pool filters on the SPATIAL mask, so a withheld year still reached the objective
    through the Ruzicka term -- and the measurement would be reading trained-on years."""
    from src.community_encoder.train_DESK.desk_training import spacetime_metric_pool
    W = 4
    pip = np.array([[0, 0, 1970], [0, 1, 1970], [0, 0, 2025], [0, 1, 2025]], dtype=np.int32)
    Xp = np.ones((4, 2), "float32")
    m_tr = np.ones((1, W), bool)
    yrs, _flat, _rows = spacetime_metric_pool(pip, Xp, np.ones(4, bool), m_tr, W,
                                              exclude_years=[1970])
    assert sorted(set(yrs)) == [2025], sorted(set(yrs))
    # and with nothing withheld the pool is unchanged
    yrs_all, _f, _r = spacetime_metric_pool(pip, Xp, np.ones(4, bool), m_tr, W)
    assert sorted(set(yrs_all)) == [1970, 2025]


def test_an_empty_holdout_list_does_not_filter_anything():
    """The default. An `np.isin` against an empty array would drop every row."""
    from src.community_encoder.train_DESK.desk_training import spacetime_metric_pool
    pip = np.array([[0, 0, 1970], [0, 1, 2025]], dtype=np.int32)
    for excl in ((), [], None if False else []):
        yrs, _f, _r = spacetime_metric_pool(pip, np.ones((2, 2), "float32"),
                                            np.ones(2, bool), np.ones((1, 4), bool), 4,
                                            exclude_years=excl)
        assert len(yrs) == 2


def test_the_temporal_holdout_targets_are_converted_to_tensors():
    """The bug that killed the first temporal-extrapolation run, in the only form a CPU can
    check. The targets are built as NUMPY in run_desk_experiment; _z_mse subtracts them from a
    model tensor. That works silently on CPU and raises on CUDA -- so a behavioural test here
    passes with the bug present (verified), and only the conversion contract is testable."""
    import torch
    from src.community_encoder.train_DESK.desk_training import device_targets
    H, W, L = 4, 5, 3
    rng = np.random.default_rng(0)
    hy = {2023: (rng.normal(size=(H, W, L)).astype("float32"), np.ones((H, W), bool))}
    out = device_targets(hy, torch.device("cpu"))
    zg, m = out[2023]
    assert torch.is_tensor(zg) and torch.is_tensor(m)
    assert zg.dtype == torch.float32 and m.dtype == torch.bool
    assert device_targets(None, torch.device("cpu")) == {}      # the unconfigured default


def test_z_mse_refuses_numpy_targets_on_every_device():
    """Turns a CUDA-only crash into an immediate, device-independent error, so this class of bug
    cannot be invisible in local testing again. Driven through the real loop, since _z_mse is a
    closure over the year index."""
    H, W, L = 16, 20, 64
    rng = np.random.default_rng(3)
    numpy_targets = {2023: (rng.normal(size=(H, W, L)).astype("float32") * 0.1,
                            np.ones((H, W), bool))}
    try:
        _tiny_train(False, epochs=2, holdout_year_targets=numpy_targets,
                    convert_targets=False)
    except TypeError as exc:
        assert "device_targets" in str(exc), exc
    else:
        raise AssertionError("numpy targets reached _z_mse without complaint")


def test_the_temporal_holdout_score_is_reported_when_configured():
    """End to end through the loop: a configured holdout year yields a real number."""
    import re
    H, W, L = 16, 20, 64
    rng = np.random.default_rng(3)
    hy = {2023: (rng.normal(size=(H, W, L)).astype("float32") * 0.1, np.ones((H, W), bool))}
    _stab, text = _tiny_train(False, epochs=2, holdout_year_targets=hy)
    vals = re.findall(r"Val\(yr-out\) ([0-9.]+|nan)", text)
    assert vals and all(v != "nan" for v in vals), (vals, text[-500:])


def test_no_temporal_holdout_reports_nan_not_a_crash():
    """The default path stays intact: nothing configured means the field is nan, not an error."""
    import re
    _stab, text = _tiny_train(False, epochs=2)
    assert re.findall(r"Val\(yr-out\) nan", text), text[-400:]


def _tiny_ema_run(tgt_extra=None, **kw):
    """Smallest train_model_ema call that still produces the diagnostic lines. Returns stdout."""
    import contextlib
    import io

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 6, 10, 12, 16
    rng = np.random.default_rng(0)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    msk = np.ones((T, H, W), bool)
    years = list(range(2020, 2026))
    m = np.ones((H, W), bool)
    ho = np.zeros((H, W), bool); ho[:, :4] = True          # a real spatial split
    tgt = {}
    for y in years[1:]:
        zg = rng.normal(size=(H, W, L)).astype("float32")
        tgt[y] = (zg, m & ~ho, m & ho, np.ones((H, W), dtype="float32"))
    if tgt_extra:
        tgt_extra(tgt, m, ho)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        D.train_model_ema(cov, msk, years, tgt,
                          _metric_dict(rng.random((H, W, 12)).astype("float32"), m, years),
                          m & ~ho, m & ho, dims, latent_dim=L,
                          ema_cfg={"earlystop_warmup": 1}, spatial_kernel=3, epochs=2, lr=1e-3,
                          weights={"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1},
                          schema=sch, dropout=0.1, **kw)
    return out.getvalue()


def _withhold(year):
    """Make ``year`` temporally withheld, the way run_desk_experiment does: train mask zeroed."""
    def apply(tgt, m, ho):
        zg, _tr, va, wg = tgt[year]
        tgt[year] = (zg, np.zeros_like(m), va, wg)
    return apply


def test_the_direction_anchor_can_be_pinned_so_the_interval_stops_moving():
    """Auto-selecting the deepest TRAINED year shortened the diagnostic interval across the
    temporal sweep (49/39/29 yr), so most of the apparent dcos decline was a shrinking chord.
    Pinning makes the interval a constant and the holdout width the only variable."""
    auto = _tiny_ema_run(tgt_extra=_withhold(2021))
    assert "rotation/direction diagnostic 2022->2025" in auto, auto   # 2021 withheld -> moved
    pinned = _tiny_ema_run(tgt_extra=_withhold(2021), direction_anchor_year=2023)
    assert "rotation/direction diagnostic 2023->2025" in pinned, pinned
    assert "trained-era CONTROL" in pinned, pinned


def test_a_pinned_anchor_that_cannot_work_fails_loudly():
    """A mis-set overlay must not silently fall back to auto-select: the whole point of pinning
    is that the interval is known, so a quiet fallback would misreport the experiment."""
    import pytest

    with pytest.raises(ValueError, match="not a supervised year"):
        _tiny_ema_run(direction_anchor_year=1900)
    # the trained-era control needs training cells; a withheld year cannot serve as one
    with pytest.raises(ValueError, match="no training cells"):
        _tiny_ema_run(tgt_extra=_withhold(2021), direction_anchor_year=2021)
    # ...but the WITHHELD anchor is allowed to be exactly that year, by design
    txt = _tiny_ema_run(tgt_extra=_withhold(2021), direction_withheld_anchor_year=2021)
    assert "WITHHELD-era direction pair 2021->2025" in txt, txt


def test_the_withheld_pair_gets_no_idw_bar_rather_than_a_rigged_one():
    """spatial_interp_dir_cos needs >=k TRAINING cells in the deep year and a withheld year has
    none, so it self-disables. Forcing a value would hand the bar that year's truth while the
    model saw none of it."""
    txt = _tiny_ema_run(tgt_extra=_withhold(2021), direction_withheld_anchor_year=2021)
    line = [l for l in txt.splitlines() if "WITHHELD-era direction pair" in l][0]
    assert "n/a" in line, line
    assert "spacetime_idw" in line, line


def test_val_mse_is_split_into_trained_era_and_withheld_era():
    """E and H are the results of the temporal sweep. Reporting only the pool forced them to be
    recovered by hand from three runs, and invited comparing the pool against a bar that covers
    only the trained years."""
    import re

    txt = _tiny_ema_run(tgt_extra=_withhold(2021))
    rows = re.findall(r"va\(pool\) ([\d.]+) gap [\d.]+x \| va\(sp\) ([\d.]+) "
                      r"va\(sp\+t\) ([\d.]+)", txt)
    assert rows, txt
    for pool, sp, spt in rows:
        pool, sp, spt = float(pool), float(sp), float(spt)
        # the pool is a cell-count-weighted mix of the two, so it must lie between them
        assert min(sp, spt) - 1e-6 <= pool <= max(sp, spt) + 1e-6, (pool, sp, spt)
    # with no temporal holdout the withheld column has nothing to report
    assert "va(sp+t) nan" in _tiny_ema_run()


def test_the_withheld_split_follows_the_train_mask_not_a_passed_list():
    """The split must be derived from what the objective actually saw. Zeroing a year's train
    mask IS how a year is withheld, so that mask -- not a separately-passed list that could drift
    from it -- has to define the reported split."""
    import re

    # tgt alone marks 2021 withheld; holdout_year_targets is deliberately NOT passed
    txt = _tiny_ema_run(tgt_extra=_withhold(2021))
    spt = re.findall(r"va\(sp\+t\) ([\d.]+|nan)", txt)
    assert spt and all(v != "nan" for v in spt), txt


def test_a_leaky_community_point_set_is_refused():
    """The focal species must never be in the community it is regressed on: Z would encode its own
    target and the circularity would be invisible, because every metric would still look
    plausible. Two independent filters keep it out today, but they key on DIFFERENT identifiers
    (an Avibase ID in avonet, an eBird code in read_community_codes), so one could silently retire
    while the other kept working. This asserts on the artifact actually being trained."""
    import pytest

    from src.community_encoder.train_DESK.desk_training import assert_focal_excluded

    clean = {"species": ["lesgol", "balori", "houspa"]}
    assert assert_focal_excluded(clean, "houfin") == 3

    with pytest.raises(ValueError, match="circular"):
        assert_focal_excluded({"species": ["lesgol", "houfin"]}, "houfin")
    # case and whitespace must not be a way past it
    with pytest.raises(ValueError, match="circular"):
        assert_focal_excluded({"species": ["lesgol", " HOUFIN "]}, "houfin")
    # a missing species list or focal code is a DIFFERENT failure and must not read as leakage
    assert assert_focal_excluded({}, "houfin") is None
    assert assert_focal_excluded(clean, "") is None


def test_stratum_weights_floor_protects_thin_strata():
    """The failure a naive rebalance invites. Full inverse-frequency hands the THINNEST stratum the
    LARGEST boost -- exactly the handful of noisy Great Plains cell-years from the late 1960s that
    cannot support an estimate. Below n_min the weight must stop rising."""
    from src.community_encoder.train_DESK.desk_training import stratum_weights
    # three strata: very thin (5), just under the floor (150), well above it (5000)
    labels = np.concatenate([np.zeros(5, int), np.ones(150, int), np.full(5000, 2)])
    w = stratum_weights(labels, n_min=200, cap=5.0)
    w_thin, w_mid, w_dense = w[0], w[10], w[-1]
    # both sub-floor strata are pinned to the SAME weight -- no extra uplift for being thinner
    assert np.isclose(w_thin, w_mid), (w_thin, w_mid)
    # and the dense stratum is downweighted relative to them
    assert w_dense < w_thin
    # full inverse-frequency would have separated them by ~sqrt(150/5)=5.5x; the floor forbids it
    assert w_thin / w_mid < 1.01


def test_stratum_weights_are_partial_not_full_correction():
    """power=0.5, not 1.0. A sqrt correction removes most of the population tilt while leaving a
    thin stratum a fraction of the pull full inverse-frequency would give it. If this ever became
    1/n, a dozen cell-years would carry the weight of thousands."""
    from src.community_encoder.train_DESK.desk_training import stratum_weights
    labels = np.concatenate([np.zeros(400, int), np.ones(40000, int)])
    w = stratum_weights(labels, n_min=100, cap=100.0)     # cap wide open, so shrinkage shows
    ratio = w[0] / w[-1]
    assert np.isclose(ratio, np.sqrt(40000 / 400), rtol=0.02), ratio   # sqrt, = 10x
    assert ratio < (40000 / 400)                                        # NOT 100x


def test_stratum_weights_cap_binds():
    """Whatever the occupancy table turns out to be, the worst-case ratio is bounded."""
    from src.community_encoder.train_DESK.desk_training import stratum_weights
    labels = np.concatenate([np.zeros(300, int), np.ones(3_000_000 // 100, int)])
    w = stratum_weights(labels, n_min=100, cap=2.0)
    assert w.max() <= 2.0 + 1e-9 and w.min() >= 0.5 - 1e-9, (w.min(), w.max())


def test_stratum_weights_are_median_normalised():
    """They compose multiplicatively with first_year_weight, so a median of 1.0 keeps the effective
    learning rate on the weighted terms unchanged when the rebalance is switched on."""
    from src.community_encoder.train_DESK.desk_training import stratum_weights
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 40, size=20000)
    w = stratum_weights(labels)
    assert abs(float(np.median(w)) - 1.0) < 0.15, float(np.median(w))


def test_weighted_pair_draw_moves_toward_balance_but_not_to_uniform():
    """The realised draw is what actually matters -- the weights are only a means. On a deliberately
    imbalanced pool the sampled stratum shares must move toward balance and stop short of it. Full
    equalisation would mean the sqrt shrinkage or the cap is not being applied, which is the
    over-fitting failure the guards exist to prevent."""
    import torch

    from src.community_encoder.train_DESK.desk_training import (
        spacetime_kernel_loss, stratum_weights)

    # 98% of the pool in one "coastal/modern" stratum, 2% in a thin "interior/early" one
    n_dense, n_thin = 9800, 200
    labels = np.concatenate([np.zeros(n_dense, int), np.ones(n_thin, int)])
    N = n_dense + n_thin
    w = torch.tensor(stratum_weights(labels, n_min=100, cap=5.0), dtype=torch.float64)

    g = torch.Generator().manual_seed(0)
    draws = torch.multinomial(w, 200_000, replacement=True, generator=g).numpy()
    got = float((draws >= n_dense).mean())
    pop = n_thin / N                       # 0.02
    assert got > 2 * pop, (got, pop)       # meaningfully rebalanced
    assert got < 0.5, got                  # but nowhere near equal shares

    # and the loss still runs with weights supplied, on a small synthetic pool
    T, H, W, L, S = 3, 4, 5, 6, 7
    z = torch.randn(T, H * W, L).reshape(T, H, W, L)
    pool_t = torch.randint(0, T, (N,))
    pool_flat = torch.randint(0, H * W, (N,))
    pool_x = torch.rand(N, S)
    out = spacetime_kernel_loss(z, pool_t, pool_flat, pool_x, num_pairs=64,
                               generator=torch.Generator().manual_seed(1), pool_w=w.float())
    assert torch.isfinite(out), out


def test_unweighted_pair_draw_is_unchanged():
    """pool_w=None must reproduce the historical uniform draw exactly, so switching the rebalance
    off restores every previously reported number."""
    import torch

    from src.community_encoder.train_DESK.desk_training import spacetime_kernel_loss
    T, H, W, L, S, N = 3, 4, 5, 6, 7, 500
    z = torch.randn(T, H, W, L)
    pool_t, pool_flat = torch.randint(0, T, (N,)), torch.randint(0, H * W, (N,))
    pool_x = torch.rand(N, S)
    a = spacetime_kernel_loss(z, pool_t, pool_flat, pool_x, num_pairs=128,
                              generator=torch.Generator().manual_seed(7))
    b = spacetime_kernel_loss(z, pool_t, pool_flat, pool_x, num_pairs=128,
                              generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_windowed_augmentation_averages_raw_counts_not_log1p():
    """The order that makes the added points the RIGHT points. X arrives log1p-transformed, so the
    augmentation must expm1 back to counts, average, and re-apply log1p. Averaging the log1p values
    would be biased low for the Poisson rate (Jensen), so the synthetic rows would describe a
    community that is systematically sparser than the one they claim to summarise."""
    from src.community_encoder.train_DESK.esk_kernel import augment_with_windowed
    raw = np.array([[0.0, 40.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    X = np.log1p(raw).astype("float32")
    pidx = np.array([[0, 0, 2000], [0, 0, 2001], [0, 0, 2002]], dtype=np.int32)
    X2, pidx2, is_synth = augment_with_windowed(X, pidx, half_width=2, min_years=2)
    assert is_synth.sum() == 3 and (~is_synth).sum() == 3
    # the centre-2001 row averages all three years
    row = X2[is_synth][np.where(pidx2[is_synth][:, 2] == 2001)[0][0]]
    right = np.log1p(raw.mean(0))                    # average counts, then transform
    wrong = np.log1p(raw).mean(0)                    # transform, then average
    assert np.allclose(row, right, atol=1e-5), (row, right)
    assert not np.allclose(row, wrong, atol=1e-3)
    assert right[1] > wrong[1]                       # log1p(mean) > mean(log1p), by Jensen


def test_windowed_augmentation_leaves_annual_points_untouched():
    """The support is WIDENED, not replaced. If the annual rows moved, annual queries would become
    the off-span ones and DESK -- which emits per-year z -- would be the thing that broke."""
    from src.community_encoder.train_DESK.esk_kernel import augment_with_windowed
    rng = np.random.default_rng(0)
    X = np.log1p(rng.poisson(0.6, (40, 6))).astype("float32")
    pidx = np.array([[0, c, 2000 + t] for c in range(8) for t in range(5)], dtype=np.int32)
    X2, pidx2, is_synth = augment_with_windowed(X, pidx, half_width=1, min_years=2)
    assert np.array_equal(X2[~is_synth], X)
    assert np.array_equal(pidx2[~is_synth], pidx)
    assert is_synth.sum() > 0


def test_windowed_augmentation_skips_singleton_windows():
    """A 'mean' over one observation is that observation. Adding it would duplicate a point rather
    than make a new region representable, and duplicates distort the landmark Gram."""
    from src.community_encoder.train_DESK.esk_kernel import augment_with_windowed
    X = np.log1p(np.array([[1.0, 2.0], [3.0, 4.0]])).astype("float32")
    # two cells, one observation each -> no window has >= 2 years
    pidx = np.array([[0, 0, 1970], [5, 5, 2000]], dtype=np.int32)
    X2, _p2, is_synth = augment_with_windowed(X, pidx, half_width=2, min_years=2)
    assert is_synth.sum() == 0 and len(X2) == 2


def test_weighting_strata_drop_the_abundance_axis():
    """Abundance is a property of the PLACE, not a sampling-bias axis, and the two purposes want
    different axes. Including it fragmented the real pool to a median of 41 rows per stratum
    (750-1087 strata over 49-73k rows), which no weighting scheme can be supported by -- the thin
    tail becomes binning artifact, and upweighting it chases noise instead of correcting bias."""
    from src.community_encoder.train_DESK.esk_kernel import spacetime_strata
    rng = np.random.default_rng(0)
    pidx = np.stack([rng.integers(0, 64, 4000), rng.integers(0, 64, 4000),
                     rng.integers(1966, 2026, 4000)], axis=1).astype(np.int32)
    X = rng.random((4000, 6)) * 10
    with_ab, _ = spacetime_strata(pidx, X, 8, 4, include_abundance=True)
    no_ab, _ = spacetime_strata(pidx, X, 8, 4, include_abundance=False)
    n_with, n_without = len(np.unique(with_ab)), len(np.unique(no_ab))
    # dropping the axis must coarsen substantially -- that is the point
    assert n_without < n_with / 2, (n_with, n_without)
    # and the coarse labelling must be a strict GROUPING of the fine one, not a different cut
    from src.community_encoder.train_DESK.esk_kernel import nests_within
    assert nests_within(with_ab, no_ab)


def test_a_floor_above_every_stratum_makes_the_correction_inert():
    """Reproduces the failure the first real run hid. With n_min above the largest stratum, every
    stratum gets the floor weight, so the spread collapses to a mild downweight of the densest
    cells with NO uplift anywhere -- and it still prints as enabled. This is why n_min is now
    derived from the occupancy table instead of guessed."""
    from src.community_encoder.train_DESK.desk_training import stratum_weights
    # realistic shape: median ~40, max ~570, as measured
    sizes = [40] * 300 + [120] * 30 + [570] * 3
    labels = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    inert = stratum_weights(labels, n_min=200, cap=5.0)
    assert inert.max() / inert.min() < 2.0, inert.max() / inert.min()
    assert np.isclose(inert.max(), 1.0, atol=0.01)          # no uplift above the median at all
    # a floor drawn from the table restores a real spread
    counts = np.bincount(labels); counts = counts[counts > 0]
    live = stratum_weights(labels, n_min=max(2, int(np.quantile(counts, 0.25))), cap=5.0)
    assert live.max() / live.min() > inert.max() / inert.min()


# ------------------- the validation kernel pool, and per-epoch instrumentation ----------------

def _val_pool_run(epochs=3, eval_kernel_pairs=512, **kw):
    """``train_model_ema`` with a real spatial split AND a val kernel pool. Returns stdout.

    Deliberately builds the train and val pools from COMPLEMENTARY masks, the way
    ``run_desk_experiment`` does, so the disjointness the tests below assert is a property of
    the construction rather than of the fixture.
    """
    import contextlib
    import io

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 6, 10, 12, 16
    rng = np.random.default_rng(0)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    msk = np.ones((T, H, W), bool)
    years = list(range(2020, 2026))
    m = np.ones((H, W), bool)
    ho = np.zeros((H, W), bool); ho[:, :4] = True
    tr_m, va_m = m & ~ho, m & ho
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32"), tr_m, va_m,
               np.ones((H, W), dtype="float32")) for y in years[1:]}
    x = rng.random((H, W, 12)).astype("float32")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        D.train_model_ema(cov, msk, years, tgt, _metric_dict(x, tr_m, years),
                          tr_m, va_m, dims, latent_dim=L,
                          ema_cfg={"earlystop_warmup": 1}, spatial_kernel=3,
                          epochs=epochs, lr=1e-3,
                          weights={"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1},
                          schema=sch, dropout=0.1,
                          val_metric_pool=_metric_dict(x, va_m, years),
                          eval_kernel_pairs=eval_kernel_pairs, **kw)
    return out.getvalue()


def test_the_val_kernel_pool_shares_no_cell_with_the_train_pool():
    """The two pools come from complementary masks, so a shared cell would mean a leak.

    A held-out cell reaching the training pool would put the evaluation data into the objective
    -- and the kernel term is exactly the term the population model consumes, so the number
    would be the one most worth trusting and the least trustworthy. Asserted on the pool builder
    rather than on a log line, because the builder is what both call sites use.
    """
    from src.community_encoder.train_DESK.desk_training import spacetime_metric_pool

    H, W = 8, 9
    rng = np.random.default_rng(3)
    n = 200
    pip = np.stack([rng.integers(0, H, n), rng.integers(0, W, n),
                    rng.integers(2000, 2006, n)], axis=1)
    Xp = rng.random((n, 5)).astype("float32")
    sup = np.ones(n, bool)
    ho = np.zeros((H, W), bool); ho[:, :4] = True
    m_tr, m_val = ~ho, ho

    _ty, tf, _tx = spacetime_metric_pool(pip, Xp, sup, m_tr, W, exclude_years=())
    _vy, vf, _vx = spacetime_metric_pool(pip, Xp, sup, m_val, W, exclude_years=())
    assert len(tf) and len(vf), (len(tf), len(vf))
    assert not (set(tf.tolist()) & set(vf.tolist())), "train and val pools share a flat cell"
    # and together they account for every supervised row -- no row silently dropped from both
    assert len(tf) + len(vf) == n
    print(f"train pool {len(tf)} / val pool {len(vf)} cell-years, disjoint")


def test_the_val_kernel_metric_never_touches_a_weight():
    """The val kernel term is a METRIC, not a loss: no gradient may reach a parameter from it.

    The plan's original check was "a step with and without the val pool gives identical
    weights". That is not testable here: this trainer is NOT bit-reproducible run to run even
    with every RNG seeded and the config unchanged -- two identical runs differ by up to ~6e-5
    on a weight after two epochs, from non-deterministic multi-threaded float32 reductions on
    CPU. (The existing ablation guard survives only because it compares `Stab` printed to four
    decimals.) An equality assertion would therefore fail for a reason that has nothing to do
    with a leak, and a tolerance loose enough to pass would be loose enough to hide one.

    So the property is checked directly instead, at the two places a leak could exist:
      * the z the metric is computed on carries NO grad_fn -- it comes from the clean re-forward
        inside ``torch.no_grad``, so there is no graph to backprop through at all;
      * the loss the metric returns carries no grad_fn either, so it cannot be added to
        anything that is stepped.
    Both are properties of the tensors, immune to float noise, and neither can hold by accident.
    """
    import contextlib
    import io

    from src.community_encoder.train_DESK import desk_training as D

    seen = []
    real = D.kernel_loss_on_pairs

    def spy(z_by_t, pool_t, pool_flat, pool_x, pairs, rank=None):
        out = real(z_by_t, pool_t, pool_flat, pool_x, pairs, rank=rank)
        seen.append((z_by_t.requires_grad, z_by_t.grad_fn is not None,
                     out.requires_grad, out.grad_fn is not None))
        return out

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 5, 8, 10, 16
    rng = np.random.default_rng(1)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    m = np.ones((H, W), bool)
    ho = np.zeros((H, W), bool); ho[:, :3] = True
    tr_m, va_m = m & ~ho, m & ho
    years = list(range(2021, 2026))
    x = rng.random((H, W, 12)).astype("float32")
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32"), tr_m, va_m,
               np.ones((H, W), dtype="float32")) for y in years[1:]}
    D.kernel_loss_on_pairs = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ema = D.train_model_ema(
                cov, np.ones((T, H, W), bool), years, tgt, _metric_dict(x, tr_m, years),
                tr_m, va_m, dims, latent_dim=L, ema_cfg={"earlystop_warmup": 0},
                spatial_kernel=3, epochs=2, lr=1e-2, seed=0,
                weights={"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1},
                schema=sch, dropout=0.1, val_metric_pool=_metric_dict(x, va_m, years),
                eval_kernel_pairs=256)
    finally:
        D.kernel_loss_on_pairs = real
    assert seen, "the kernel metric was never evaluated -- the fixture proves nothing"
    for z_rg, z_graph, l_rg, l_graph in seen:
        assert not z_graph, "the z the metric reads carries a graph: a leak is possible"
        assert not l_graph, "the metric's loss carries a graph and could be backpropagated"
        assert not z_rg and not l_rg, (z_rg, l_rg)
    # The complementary half -- that the training loss itself is unmoved -- is
    # test_wiring_the_val_pool_does_not_move_the_training_loss below.
    print(f"kernel metric evaluated {len(seen)}x, always off-graph")


def test_wiring_the_val_pool_does_not_move_the_training_loss():
    """``Stab`` must be identical with and without the val kernel pool.

    Stab is the training loss, computed before the eval re-forward, so it is the exact quantity
    a gradient leak from the val term would move -- and unlike a weight tensor it is printed to
    four decimals, which is coarse enough to be stable under this trainer's float
    non-determinism and fine enough that a leak at lr=1e-3 would show.
    """
    on, _ = _tiny_train(False)
    off, _ = _tiny_train(False)
    assert on == off, (on, off)          # the baseline is stable at this precision
    txt_with = _val_pool_run(selection_metric="val_kernel")
    stab_with = re.findall(r"Stab (\d+\.\d+)", txt_with)
    txt_plain = _val_pool_run(selection_metric="val_zmse")
    stab_plain = re.findall(r"Stab (\d+\.\d+)", txt_plain)
    assert stab_with and stab_with == stab_plain, (stab_with, stab_plain)
    print(f"training loss unchanged by the val kernel metric: {stab_with}")


def test_the_kernel_metric_uses_fixed_pairs_so_epochs_are_comparable():
    """Two evaluations of the SAME model must give the same kernel number.

    The training term redraws its pairs every step, which is right for a gradient and wrong for
    a metric an epoch is selected on: with a fresh draw, consecutive epochs differ by the dice
    as well as by the model, and the argmin over 500 epochs is partly a coin flip. This is the
    property that makes best-epoch selection mean anything.
    """
    from src.community_encoder.train_DESK.desk_training import (fixed_kernel_pairs,
                                                               kernel_loss_on_pairs,
                                                               spacetime_kernel_loss)
    rng = np.random.default_rng(5)
    T, H, W, L, S, N = 4, 5, 6, 8, 7, 60
    z = torch.tensor(rng.normal(size=(T, H, W, L)).astype("float32"))
    pt = torch.tensor(rng.integers(0, T, N))
    pf = torch.tensor(rng.integers(0, H * W, N))
    px = torch.tensor(rng.random((N, S)).astype("float32"))

    pairs = fixed_kernel_pairs(N, 1024, seed=0)
    a = float(kernel_loss_on_pairs(z, pt, pf, px, pairs))
    b = float(kernel_loss_on_pairs(z, pt, pf, px, pairs))
    assert a == b, (a, b)                              # same pairs, same model -> same number
    # and the same seed reproduces the pair set, so a rerun of a run is comparable to it
    assert np.array_equal(pairs, fixed_kernel_pairs(N, 1024, seed=0))
    assert not np.array_equal(pairs, fixed_kernel_pairs(N, 1024, seed=1))
    # a redrawing loss on the same model does NOT agree with itself -- the thing being avoided
    g = torch.Generator().manual_seed(0)
    c = float(spacetime_kernel_loss(z, pt, pf, px, num_pairs=1024, generator=g))
    d = float(spacetime_kernel_loss(z, pt, pf, px, num_pairs=1024, generator=g))
    assert c != d, "the training term is supposed to redraw; the fixture is not exercising it"
    # too few points to pair is UNAVAILABLE, not zero
    assert fixed_kernel_pairs(1, 100, 0) is None
    print("fixed pairs make the kernel metric deterministic across epochs")


def test_the_trajectory_jsonl_holds_every_epoch_and_names_the_best(tmp_path):
    """Every trajectory analysed so far was recovered by regex-parsing a job log.

    That works only while the log survives and the print format holds, and it cannot record a
    quantity that was never printed. The JSONL is the artifact the sweep reads, so it must carry
    one row per epoch and the recorded best epoch must be the argmin of the selected column --
    not of some other column that happens to correlate.
    """
    path = tmp_path / "sub" / "train_trajectory.jsonl"
    txt = _val_pool_run(trajectory_path=str(path), selection_metric="val_kernel")
    rows = [json.loads(l) for l in open(path)]
    assert [r["epoch"] for r in rows] == [1, 2, 3], [r["epoch"] for r in rows]
    for r in rows:
        for key in ("kernel_val", "kernel_val_sp", "kernel_train", "zmse_val", "lr",
                    "half_life", "epoch_seconds", "selection_metric"):
            assert key in r, key
        assert r["selection_metric"] == "val_kernel"
    # the restored epoch must be the argmin of the column that was selected on, over the epochs
    # eligible for selection (earlystop_warmup=1 excludes epoch 1)
    m = re.search(r"restored best epoch (\d+)", txt)
    assert m, txt
    eligible = [r for r in rows if r["epoch"] > 1]
    want = min(eligible, key=lambda r: r["kernel_val"])["epoch"]
    assert int(m.group(1)) == want, (m.group(1), [(r["epoch"], r["kernel_val"]) for r in rows])
    print(f"trajectory holds every epoch; best epoch {want} is the argmin of kernel_val")


def test_selecting_on_the_kernel_can_pick_a_different_epoch_than_zmse(tmp_path):
    """The two metrics must be able to disagree, and the log must say which one was used.

    On a real 500-epoch run they do: val z-MSE was best at 109 and 10% worse by 500, while the
    kernel loss was still falling at 500. If the code could only ever agree with z-MSE the
    switch would be decorative, and the disagreement -- which the plan calls a finding -- would
    be unobservable.
    """
    txt_k = _val_pool_run(trajectory_path=str(tmp_path / "k.jsonl"),
                          selection_metric="val_kernel")
    txt_z = _val_pool_run(trajectory_path=str(tmp_path / "z.jsonl"),
                          selection_metric="val_zmse")
    assert "epoch selection on val_kernel" in txt_k
    assert "epoch selection on val_zmse" in txt_z
    rows = [json.loads(l) for l in open(tmp_path / "k.jsonl")]
    elig = [r for r in rows if r["epoch"] > 1]
    best_k = min(elig, key=lambda r: r["kernel_val"])["epoch"]
    best_z = min(elig, key=lambda r: r["zmse_val"])["epoch"]
    assert re.search(rf"restored best epoch {best_k}\b", txt_k), txt_k
    assert re.search(rf"restored best epoch {best_z}\b", txt_z), txt_z
    # both columns are present in both runs, so a run under one metric stays comparable with a
    # run made under the other
    for p in ("k.jsonl", "z.jsonl"):
        r0 = json.loads(open(tmp_path / p).readline())
        assert np.isfinite(r0["kernel_val"]) and np.isfinite(r0["zmse_val"])
    print(f"kernel picked ep {best_k}, z-MSE picked ep {best_z}; both logged either way")


def test_selecting_on_a_kernel_that_does_not_exist_is_refused():
    """``selection_metric='val_kernel'`` with val cells but no pool would select on NaN.

    Every comparison against NaN is False, so nothing would ever beat the initial best and the
    run would silently keep the warmup epoch's weights -- a broken run that finishes, reports a
    number, and looks like the others.

    The guard is conditioned on there BEING validation cells. A no-holdout production retrain has
    none by design and can select on no metric at all; raising there would make val_kernel
    unusable as the committed default, since the one run the sweep exists to produce would refuse
    to start. See test_a_no_holdout_run_selects_on_nothing_without_failing.
    """
    import contextlib
    import io

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 4, 6, 7, 16
    rng = np.random.default_rng(0)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    m = np.ones((H, W), bool)
    years = list(range(2022, 2026))
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32"), m, m,
               np.ones((H, W), dtype="float32")) for y in years[1:]}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            D.train_model_ema(cov, np.ones((T, H, W), bool), years, tgt,
                              _metric_dict(rng.random((H, W, 9)).astype("float32"), m, years),
                              m, m, dims, latent_dim=L, ema_cfg={}, spatial_kernel=3,
                              epochs=2, schema=sch, dropout=0.1,
                              selection_metric="val_kernel")
        raise AssertionError("selecting on an unwired kernel metric must be refused")
    except ValueError as exc:
        assert "nothing to select on" in str(exc), str(exc)
    print("selecting on a missing kernel metric is refused, not silently NaN")


def test_stop_at_epoch_leaves_the_learning_rate_schedule_untouched(tmp_path):
    """Lowering ``epochs`` is NOT the same as stopping early, and the difference is every step.

    ``_warmup_cosine`` takes the BUDGET: with ``epochs=500`` the anneal reaches its floor at
    500, so a run stopped at 300 has seen a different LR at every one of those 300 steps than a
    run configured with ``epochs=300``. The production retrain has no validation set and must
    reproduce the model the sweep selected, so it has to stop at the chosen epoch while keeping
    the schedule the sweep ran under.
    """
    from src.community_encoder.train_DESK.desk_training import _warmup_cosine

    full = _warmup_cosine(10, 2, 0.05)
    short = _warmup_cosine(4, 2, 0.05)
    assert [round(full(e), 6) for e in range(4)] != [round(short(e), 6) for e in range(4)], \
        "the fixture must exercise a schedule that a shorter budget actually changes"

    a = _val_pool_run(trajectory_path=str(tmp_path / "a.jsonl"),
                      selection_metric="val_kernel", epochs=6, warmup_epochs=1,
                      min_lr_frac=0.05)
    b = _val_pool_run(trajectory_path=str(tmp_path / "b.jsonl"),
                      selection_metric="val_kernel", epochs=6, warmup_epochs=1,
                      min_lr_frac=0.05, stop_at_epoch=3)
    lr_a = [r["lr"] for r in map(json.loads, open(tmp_path / "a.jsonl"))]
    lr_b = [r["lr"] for r in map(json.loads, open(tmp_path / "b.jsonl"))]
    assert len(lr_a) == 6 and len(lr_b) == 3, (len(lr_a), len(lr_b))
    assert lr_a[:3] == lr_b, (lr_a[:3], lr_b)          # identical LR at every step up to the stop
    assert "stop_at_epoch=3 reached" in b
    print(f"stop_at_epoch truncates the run, not the schedule: {lr_b} == {lr_a[:3]}")


def test_a_run_with_no_validation_set_keeps_its_final_weights():
    """The production retrain holds nothing out, so ``best`` is never set.

    The code restored ``best`` unconditionally, so this path failed on a ``None`` -- a
    no-holdout run could not finish at all, which is the one run the whole sweep exists to
    produce. Keeping the FINAL weights is the only defensible answer, and the log has to say
    that no epoch was selected: a run with no measured skill must not be reported as if it had
    some.
    """
    import contextlib
    import io

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 4, 6, 7, 16
    rng = np.random.default_rng(2)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    m = np.ones((H, W), bool)
    empty = np.zeros((H, W), bool)                    # NO validation cells anywhere
    years = list(range(2022, 2026))
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32"), m, empty,
               np.ones((H, W), dtype="float32")) for y in years[1:]}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        model, _ema, info = D.train_model_ema(
            cov, np.ones((T, H, W), bool), years, tgt,
            _metric_dict(rng.random((H, W, 9)).astype("float32"), m, years),
            m, empty, dims, latent_dim=L, ema_cfg={"earlystop_warmup": 0},
            spatial_kernel=3, epochs=3, schema=sch, dropout=0.1, return_info=True)
    txt = out.getvalue()
    assert "keeping the FINAL weights from epoch 3" in txt, txt
    assert "no validation cells" in txt, txt
    assert info["restored_best"] is False and info["best_epoch"] == 3, info
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print("a no-holdout run finishes and keeps its final weights")


def test_return_info_is_behind_a_flag_so_the_arity_cannot_change_silently():
    """A silent arity change is a failure mode this file has been bitten by.

    ``spacetime_metric_pool`` puts its fourth element behind ``return_pidx`` for exactly this
    reason. The default 2-tuple keeps every existing call site working; the 3-tuple is opt-in.
    """
    import contextlib
    import io

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 3, 5, 6, 16
    rng = np.random.default_rng(4)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    m = np.ones((H, W), bool)
    years = list(range(2023, 2026))
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32"), m, m,
               np.ones((H, W), dtype="float32")) for y in years[1:]}
    args = (cov, np.ones((T, H, W), bool), years, tgt,
            _metric_dict(rng.random((H, W, 9)).astype("float32"), m, years), m, m, dims)
    kw = dict(latent_dim=L, ema_cfg={"earlystop_warmup": 0}, spatial_kernel=3, epochs=1,
              schema=sch, dropout=0.1)
    with contextlib.redirect_stdout(io.StringIO()):
        assert len(D.train_model_ema(*args, **kw)) == 2
        assert len(D.train_model_ema(*args, return_info=True, **kw)) == 3
    print("return_info is opt-in; the default arity is unchanged")


def test_the_stabilizing_target_excludes_the_buffer_and_any_thinned_blocks():
    """The buffer was excluded from the metric pool and the fit, but NOT from the STABILIZING
    term -- which carries most of the loss.

    That silently voided the buffer's guarantee: a held-out cell's convolutional receptive field
    reaches ``kernel//2`` cells away, and those cells were supervised, so every held-out score
    was measured against a model trained right up to the block edge. It is also what makes a
    ``train_frac`` axis mean anything -- thinning the metric pool alone would leave the dominant
    term reading every cell, and the data-amount trajectory would be mostly flat by
    construction.
    """
    from src.community_encoder.train_DESK.desk_training import _prepare_trend_targets

    H, W = 6, 8
    ho = np.zeros((H, W), bool); ho[:, :2] = True
    buf = np.zeros((H, W), bool); buf[:, 2] = True
    drop = np.zeros((H, W), bool); drop[0, 5] = True
    calls = {}

    class _FakeZ:
        pass

    # Patch the two things _prepare_trend_targets reaches out for, so the mask arithmetic can be
    # tested without an ESK basis on disk.
    import src.community_encoder.train_DESK.desk_training as D
    import src.community_encoder.train_DESK.esk_kernel as K
    n = H * W
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pidx = np.stack([rr.ravel(), cc.ravel(), np.full(n, 2020)], axis=1)
    saved_proj, saved_dir = K.project_points_to_z, D.target_points_dir if hasattr(
        D, "target_points_dir") else None
    try:
        K.project_points_to_z = lambda X, zd, ld: np.zeros((X.shape[0], ld), "float32")
        D.load_point_set = lambda _d: (np.ones((n, 3), "float32"), pidx,
                                       np.ones(n, "float32"), np.ones(n, bool))
        cfg = {"target": {"points_dir": "/nonexistent"}, "trend": {"points_dir": "/nonexistent"}}
        out = _prepare_trend_targets(cfg, "/nonexistent", 4, ho, points_dir="/nonexistent",
                                     exclude=buf | drop)
    finally:
        K.project_points_to_z = saved_proj
    _zg, tr, va, _wg = out[2020]
    assert not (tr & ho).any(), "held-out cells reached the stabilizing train mask"
    assert not (tr & buf).any(), "BUFFER cells reached the stabilizing train mask"
    assert not (tr & drop).any(), "train_frac-dropped cells reached the stabilizing train mask"
    assert (va == ho).all(), "the val mask must be exactly the holdout"
    # the excluded cells go to NEITHER side -- they are not evaluation data either
    assert not (va & buf).any() and not (va & drop).any()
    assert int(tr.sum()) == int(((~ho) & (~buf) & (~drop)).sum())
    print(f"stabilizing train mask: {int(tr.sum())} cells, buffer and thinned blocks excluded")


def test_eval_every_above_one_runs_and_records_only_measured_epochs(tmp_path):
    """``eval_every > 1`` raised UnboundLocalError on the first un-evaluated epoch.

    ``vs``, ``ts``, ``rotP`` and every other reported quantity are bound only inside the eval
    branch, while the log line and the ratio columns that read them sat outside it -- so the
    knob the config documents as "amortizes the clean eval re-forward" could not be used at
    all. It is a loud crash rather than a wrong number, but it is on the Tier-3 list and would
    have failed on the cluster, per run.

    Pre-binding those names would have been worse than the crash: the eval-only values persist
    across iterations, so an un-evaluated epoch would have printed and RECORDED the previous
    eval's numbers under its own epoch number -- a stale value that reads as a fresh
    measurement, which is the failure mode this project has been burned by most. So an
    un-evaluated epoch reports only what was actually measured (the training losses) and gets
    no trajectory row and no shot at being selected.
    """
    for ee, want_rows in ((1, [1, 2, 3, 4]), (2, [2, 4]), (3, [3, 4])):
        path = tmp_path / f"e{ee}.jsonl"
        txt = _val_pool_run(epochs=4, eval_every=ee, trajectory_path=str(path),
                            selection_metric="val_kernel")
        rows = [json.loads(l) for l in open(path)]
        assert [r["epoch"] for r in rows] == want_rows, (ee, [r["epoch"] for r in rows])
        assert all(r["eval_every"] == ee for r in rows), ee
        # every recorded row carries a real measurement, never a carried-over one
        for r in rows:
            assert np.isfinite(r["kernel_val"]) and np.isfinite(r["zmse_val"]), (ee, r["epoch"])
        # un-evaluated epochs say so instead of reprinting stale metrics
        assert txt.count("no eval this epoch") == 4 - len(want_rows), (ee, txt)
        # and the selected epoch is one that was actually evaluated
        m = re.search(r"restored best epoch (\d+)", txt)
        assert m and int(m.group(1)) in want_rows, (ee, m.group(1) if m else None, want_rows)
    print("eval_every>1 works and records only measured epochs")


def test_a_no_holdout_run_selects_on_nothing_without_failing():
    """With no validation cells, val_kernel must be a no-op rather than an error.

    This is the production retrain: holdout_frac=0, nothing held out, so neither metric exists.
    It keeps its final weights and takes its stopping point from stop_at_epoch, chosen on the
    sweep grid. Making the committed default val_kernel would otherwise have broken exactly the
    run the whole sweep exists to produce -- and the failure would have appeared only at the very
    end, after every grid run had been paid for.

    The distinction that makes this safe is between "no validation set" (deliberate) and "a
    validation set whose kernel pool did not build" (a wiring bug). The second still raises.
    """
    import contextlib
    import io

    from src.community_encoder.train_DESK import desk_training as D

    sch = _schema()
    dims = [s["dim"] for s in sch["streams"]]
    T, H, W, L = 4, 6, 7, 16
    rng = np.random.default_rng(2)
    cov = rng.normal(size=(T, H, W, sch["total_dim"])).astype("float32")
    m = np.ones((H, W), bool)
    empty = np.zeros((H, W), bool)
    years = list(range(2022, 2026))
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32"), m, empty,
               np.ones((H, W), dtype="float32")) for y in years[1:]}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _model, _ema, info = D.train_model_ema(
            cov, np.ones((T, H, W), bool), years, tgt,
            _metric_dict(rng.random((H, W, 9)).astype("float32"), m, years), m, empty, dims,
            latent_dim=L, ema_cfg={"earlystop_warmup": 0}, spatial_kernel=3, epochs=3,
            schema=sch, dropout=0.1, selection_metric="val_kernel", stop_at_epoch=2,
            return_info=True)
    txt = out.getvalue()
    assert "no validation cells, so no epoch can be selected" in txt, txt
    assert "stop_at_epoch" in txt, "the log must say where the stopping point has to come from"
    assert info["restored_best"] is False and info["best_epoch"] == 2, info
    assert info["epochs_run"] == 2 and info["epochs_budget"] == 3, info
    print("a no-holdout run under val_kernel finishes and keeps its final weights")


def test_the_committed_default_selects_on_the_kernel():
    """The config default must be the metric the downstream consumes.

    Not contingent on a measurement: two selection metrics cannot be ranked by outcome, since
    selecting on one always yields the better value of that one. The population model reads Z
    through learned linear weights, so its covariance IS the similarity -- which settles it. An
    earlier version of the config comment made the default conditional on a pending run; that
    conflated the metric choice with the separate question of the epoch BUDGET.
    """
    import json as _json
    import pathlib

    cfg = _json.loads((pathlib.Path(__file__).resolve().parents[1] / "config"
                       / "esk_desk_config.json").read_text(encoding="utf-8"))
    assert cfg["desk"]["selection_metric"] == "val_kernel", cfg["desk"]["selection_metric"]
    note = cfg["desk"]["_selection_comment"]
    assert "NOT contingent on a measurement" in note
    assert "cannot be ranked by outcome" in note
    assert "stop_at_epoch" in note, \
        "the comment must say how a no-validation run gets its stopping point"
    print("committed default selects on the kernel term")


def test_selection_smoothing_is_off_by_default_and_only_moves_selection(tmp_path):
    """A trailing median over the selection signal, applied to SELECTION only.

    The argmin of a noisy series is not a property of the model. Measured on the first real
    30-epoch run: kernel_val swung 2.9x between adjacent epochs while the LR was near peak, and
    the epoch selected sat 24% below the median of the converged region with its immediate
    neighbour 3.1x higher -- an isolated spike. Ranking 17 configurations by each one's own best
    value then partly ranks which got the luckier evaluation.

    OFF by default on purpose: the noise is LR-driven and largely self-correcting (once the
    cosine took the LR below ~4e-4 the spread fell to 1.017x), so this must not be switched on by
    assumption. What it must never do is alter what is LOGGED -- the per-epoch trajectory stays
    raw, so a smoothed run's numbers remain comparable with an unsmoothed one's even though its
    chosen epoch is not.
    """
    import pathlib

    cfg = json.loads((pathlib.Path(__file__).resolve().parents[1] / "config"
                      / "esk_desk_config.json").read_text(encoding="utf-8"))
    assert cfg["desk"]["selection_smooth"] == 0, "must not be enabled by assumption"

    raw_path, sm_path = tmp_path / "raw.jsonl", tmp_path / "sm.jsonl"
    txt_raw = _val_pool_run(epochs=8, trajectory_path=str(raw_path),
                            selection_metric="val_kernel", selection_smooth=0)
    txt_sm = _val_pool_run(epochs=8, trajectory_path=str(sm_path),
                           selection_metric="val_kernel", selection_smooth=3)
    raw = [json.loads(l) for l in open(raw_path)]
    sm = [json.loads(l) for l in open(sm_path)]
    # The logged trajectories match to within this trainer's float noise -- NOT exactly. Two
    # identical runs differ by ~1e-8 from non-deterministic CPU float32 reductions (see
    # test_the_val_kernel_metric_never_touches_a_weight), so an equality assertion here would
    # fail for a reason unrelated to smoothing. The tolerance is orders of magnitude below the
    # 2.9x epoch-to-epoch swing this feature exists to address, so it cannot hide a smoothed
    # series being logged in place of the raw one.
    assert np.allclose([r["kernel_val"] for r in raw], [r["kernel_val"] for r in sm],
                       rtol=1e-4, atol=0), (raw[0]["kernel_val"], sm[0]["kernel_val"])
    # The decisive within-run check: the smoothed run's own log is RAW. A trailing median is
    # monotone-ish and would visibly flatten the series, so if the logged values were smoothed
    # they could not still equal the unsmoothed run's.
    for txt in (txt_raw, txt_sm):
        assert "restored best epoch" in txt, txt
    # and the unsmoothed selection is the plain argmin over the eligible epochs
    elig = [r for r in raw if r["epoch"] > 1]
    want = min(elig, key=lambda r: r["kernel_val"])["epoch"]
    assert int(re.search(r"restored best epoch (\d+)", txt_raw).group(1)) == want
    print("selection smoothing is opt-in and leaves the recorded trajectory untouched")


def test_the_best_epoch_is_the_argmin_not_the_first_epoch_within_an_epsilon(tmp_path):
    """``min_delta`` is absolute, and the two selection metrics differ by ~30x in scale.

    1e-4 is 0.05% of a val z-MSE (~0.2) and 1.4% of a val kernel (~0.007). At the old hardcoded
    1e-4, moving selection to the kernel redefined "best epoch" as "first epoch within 1e-4 of
    the best": 11 of 17 stage-1 runs recorded a best_epoch that was not their argmin and a best
    value that was not their minimum, and the cross-configuration ranking built on those values
    was comparing the selection epsilon as much as the configurations.

    0 is not merely a safer default, it is the only correct one while ``patience`` equals
    ``epochs``: early stopping cannot fire, so min_delta's sole remaining effect is to prevent
    the best epoch from being the best epoch.
    """
    path = tmp_path / "t.jsonl"
    txt = _val_pool_run(epochs=8, trajectory_path=str(path), selection_metric="val_kernel")
    rows = [json.loads(l) for l in open(path)]
    elig = [r for r in rows if r["epoch"] > 1]          # earlystop_warmup=1
    want = min(elig, key=lambda r: r["kernel_val"])["epoch"]
    got = int(re.search(r"restored best epoch (\d+)", txt).group(1))
    assert got == want, (got, want, [(r["epoch"], r["kernel_val"]) for r in rows])

    # and an explicit epsilon large against this metric's scale still shifts it, which is the
    # mechanism that caused the bug -- so it stays observable rather than removed
    txt2 = _val_pool_run(epochs=8, trajectory_path=str(tmp_path / "u.jsonl"),
                         selection_metric="val_kernel", min_delta=1.0)
    got2 = int(re.search(r"restored best epoch (\d+)", txt2).group(1))
    assert got2 <= got, (got2, got)
    print(f"best epoch {got} is the argmin; a large min_delta still moves it to {got2}")


def test_independent_draws_give_a_real_error_bar_on_the_selected_metric(tmp_path):
    """A single draw yields a number with no error bar, which is why stage 1 was unreadable.

    17 configurations spread over 8% of the held-out kernel and nothing could say whether that
    was above the sampling noise of the kernel estimate itself. With several independent draws the
    mean is the selection signal and the spread across draws is the estimator's noise floor, so
    the comparison that decides whether a sweep could resolve anything becomes available.

    Asserts the spread is non-zero (it is a real measurement, not a placeholder) and that it
    SHRINKS with more pairs at the expected rate -- if it did not, the number would not be
    sampling error and averaging draws would not help.
    """
    def sd_at(pairs, draws):
        p = tmp_path / f"t{pairs}_{draws}.jsonl"
        _val_pool_run(epochs=3, trajectory_path=str(p), selection_metric="val_kernel",
                      eval_kernel_pairs=pairs, eval_kernel_draws=draws)
        rows = [json.loads(l) for l in open(p)]
        assert all(r["kernel_val_draws"] == draws for r in rows)
        return np.mean([r["kernel_val_sd"] for r in rows]), \
            np.mean([r["kernel_val"] for r in rows])

    sd_small, mean_small = sd_at(256, 6)
    sd_big, mean_big = sd_at(4096, 6)
    assert sd_small > 0, "the spread must be a real measurement"
    # 16x the pairs should cut the standard deviation ~4x (it falls as 1/sqrt(P)). Loose bounds:
    # 6 draws estimate a std to only +-32%, so this checks the scaling law, not a constant.
    ratio = sd_small / sd_big
    assert 2.0 < ratio < 8.0, (sd_small, sd_big, ratio)
    # and the MEAN is stable across pair counts -- more pairs reduce variance, not shift the
    # estimand. If it moved, the metric would not be comparable across configurations either.
    assert abs(mean_big - mean_small) / mean_small < 0.15, (mean_small, mean_big)
    print(f"sd {sd_small:.5f} -> {sd_big:.5f} for 16x pairs ({ratio:.1f}x), mean stable")


def test_one_draw_reproduces_the_previous_single_draw_behaviour(tmp_path):
    """``eval_kernel_draws=1`` must be the old code path exactly, so old runs stay comparable."""
    p = tmp_path / "one.jsonl"
    _val_pool_run(epochs=2, trajectory_path=str(p), selection_metric="val_kernel",
                  eval_kernel_draws=1)
    rows = [json.loads(l) for l in open(p)]
    assert all(r["kernel_val_draws"] == 1 for r in rows)
    # A single draw has no spread; it must report 0, not nan -- nan would read as "unavailable"
    # when the truth is "unmeasurable from one sample".
    assert all(r["kernel_val_sd"] == 0.0 for r in rows), [r["kernel_val_sd"] for r in rows]
    print("a single draw reports zero spread, not nan")


def test_the_three_val_pools_draw_from_independent_seed_streams():
    """pool/sp/spt previously shared one seed, so they were not independent draws.

    They differed only because the pools had different lengths, which is not independence -- and a
    spread computed across them would have understated the noise. Checked on the helper rather
    than through a run, because the property is a property of the seeding.
    """
    from src.community_encoder.train_DESK.desk_training import (POOL_SEED_OFFSET,
                                                                kernel_pair_draws)
    n, pairs, seed = 5000, 64, 7
    per_pool = {}
    for label, off in POOL_SEED_OFFSET.items():
        d = kernel_pair_draws(n, pairs, seed + 104729 * (off + 1), 3)
        per_pool[label] = d
        # draws WITHIN a pool are independent of each other
        assert not np.array_equal(d[0], d[1]) and not np.array_equal(d[1], d[2]), label
    # and no two pools share a draw
    flat = [(lab, i, tuple(map(tuple, d))) for lab, ds in per_pool.items()
            for i, d in enumerate(ds)]
    assert len({f for _l, _i, f in flat}) == len(flat), "two pools share a pair set"
    # too few points is UNAVAILABLE for the whole set, not a partially-populated list
    assert kernel_pair_draws(1, 64, 0, 4) is None
    print(f"{len(flat)} pair sets across {len(per_pool)} pools, all distinct")


def test_the_rank_curve_measures_what_a_truncating_downstream_would_get(tmp_path):
    """Selection stays full-rank; the curve reports the ranks the downstream might adopt.

    The population model truncates positionally to the first M components, so the rank-M kernel is
    the covariance of the GP it fits -- and that is not the rank-64 number selection runs on. The
    curve is reported for several ranks on purpose: M is 24 today and plausibly 32 later, so
    keying any decision on one rank would bake in a moving target.

    Also asserts the curve improves with rank, which is the sanity condition: adding components
    cannot make the kernel approximation worse if they carry any signal at all.
    """
    p = tmp_path / "ranks.jsonl"
    txt = _val_pool_run(epochs=2, trajectory_path=str(p), selection_metric="val_kernel",
                        eval_kernel_ranks=(4, 8, 16), eval_kernel_draws=2)
    row = [json.loads(l) for l in open(p)][-1]
    for q in ("ema", "raw"):
        vals = [row[f"kernel_val_{q}_r{r}"] for r in (4, 8, 16)]
        assert all(np.isfinite(v) for v in vals), (q, vals)
        assert vals[0] >= vals[1] >= vals[2], f"{q} curve not monotone in rank: {vals}"
    # the full-rank ema value is the one selected on, and matches the pooled kernel column
    assert row["kernel_val_ema_r16"] == pytest.approx(row["kernel_val"], rel=1e-9)
    # z_raw is reported alongside z_ema because only z_ema is supervised while the cube exports
    # z_raw -- the gap is the point, so both must be present and they must not be aliases
    assert row["kernel_val_raw_r16"] != row["kernel_val_ema_r16"]
    assert "restored best epoch" in txt
    print("rank curve monotone on both z_ema and z_raw; full rank matches the selected value")


def test_the_pool_is_not_scored_twice_when_no_year_is_withheld():
    """With no withheld years `sp` IS `pool`; scoring both was a quarter of the eval wasted.

    Previously they also shared pair indices, so the log printed the two values identically -- a
    reader could not tell an alias from a coincidence. Aliasing makes the redundancy explicit and
    pays for the extra independent draws.
    """
    txt = _val_pool_run(epochs=1, selection_metric="val_kernel", eval_kernel_draws=2)
    assert "no withheld years, so sp == pool; scoring it once" in txt, txt
    # exactly one val pool is prepared, not two identical ones
    assert txt.count("val kernel pool (pool):") == 1
    assert "val kernel pool (sp):" not in txt
    print("pool scored once when sp is identical to it")


def test_the_degenerate_pair_branch_returns_zero_instead_of_raising():
    """``_pair_kernel_loss``'s all-invalid branch referenced a name not in its scope.

    It said ``z_pred.device`` -- a leftover from ``true_kernel_loss``, where the argument really
    is called ``z_pred`` -- so instead of returning 0 it raised NameError. Unreachable only
    because a real community is never all-zero, which is exactly the kind of latent break that
    surfaces the first time a pool is filtered down to nothing.
    """
    from src.community_encoder.train_DESK.desk_training import _pair_kernel_loss
    zeros = torch.zeros(4, 3)
    out = _pair_kernel_loss(torch.randn(4, 5), torch.randn(4, 5), zeros, zeros)
    assert float(out) == 0.0
    assert out.requires_grad
    print("a degenerate pair set yields 0, not NameError")
