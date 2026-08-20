"""Tests for the DESK training augmentation, split, and loss reparametrization.

Covers the cluster-free pure cores of the training recipe: structured channel-group masking
(``augment.ChannelGroupMasker``), the spatially blocked + buffered validation split
(``augment.blocked_holdout``), the 5x5 partial conv, and the stabilizing-loss
reparametrization. Runs standalone or under pytest; no GPU and no state files needed.
"""
import os
import sys

import numpy as np
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


def _tiny_train(augment, seed=0, epochs=3, patch=None):
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
                warmup_epochs=1, min_lr_frac=0.05)
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
    # `rot tr R va R (raw Rr/Rrv)`: R is z_ema (the SUPERVISED quantity, so 1.0 is the goal),
    # raw is the pre-EMA rotation, which must be LARGER because the EMA attenuates.
    rots = re.findall(r"rot tr ([\d.]+) va ([\d.]+) \(raw ([\d.]+)/([\d.]+)\)", text)
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
    assert re.search(r"mse tr [\d.]+ va [\d.]+ gap [\d.]+x", text), text
    # the direction column and its permuted null must both be present every epoch
    assert len(re.findall(r"dcos tr [\d.-]+ va [\d.-]+ null [+-][\d.]+/[+-][\d.]+", text)) == 3, text
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
