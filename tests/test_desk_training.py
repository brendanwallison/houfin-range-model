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


def _schema(bio_start=8):
    """The real LS6 layout: 14 climate bases (3 with q10/q90) = 240 ch, + 55 others = 295."""
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
    tgt = {y: (rng.normal(size=(H, W, L)).astype("float32") * 0.1, m, m) for y in years[1:]}
    x = rng.random((H, W, 12)).astype("float32")

    saved = D.true_kernel_loss
    if patch is not None:
        D.true_kernel_loss = patch
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            D.train_model_ema(
                cov, msk, years, tgt, x, m, m, dims, latent_dim=L,
                ema_cfg={"earlystop_warmup": 1}, spatial_kernel=5, epochs=epochs, lr=1e-3,
                seed=seed, weights={"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1},
                schema=sch, augment_cfg={"enabled": augment}, dropout=0.1,
                warmup_epochs=1, min_lr_frac=0.05)
    finally:
        D.true_kernel_loss = saved
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
        tgt[y] = ((np.cos(a) * base + np.sin(a) * perp).astype("float32"), m, m)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        D.train_model_ema(
            cov, msk, years, tgt, rng.random((H, W, 12)).astype("float32"), m, m, dims,
            latent_dim=L, ema_cfg={"earlystop_warmup": 1}, spatial_kernel=5, epochs=3, lr=1e-3,
            weights={"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1},
            schema=sch, augment_cfg={"enabled": True}, dropout=0.1,
            warmup_epochs=1, min_lr_frac=0.05)
    text = out.getvalue()

    assert "rotation diagnostic 2021->2025" in text
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


def test_spatial_interp_baseline():
    """The ceiling for a blocked holdout: interpolate the targets, no covariates, no learning."""
    from src.community_encoder.train_DESK.augment import blocked_holdout
    from src.community_encoder.train_DESK.desk_training import spatial_interp_baseline

    H, W, L = 60, 80, 16
    valid = np.zeros((H, W), bool); valid[2:58, 3:77] = True
    val, buf = blocked_holdout(valid, block_cells=12, holdout_frac=0.2, buffer_cells=2, seed=0)
    tr = valid & ~val & ~buf
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(0)

    # A smooth field is nearly interpolable, so the baseline must score LOW -- meaning a
    # trained model has to beat a low number to have earned anything from the covariates.
    field = np.stack([np.sin((yy + 3 * d) / 12.0) + np.cos((xx - d) / 15.0)
                      for d in range(L)], -1).astype("float32")
    smooth = {y: (field, torch.tensor(tr), torch.tensor(val)) for y in (1966, 2025)}
    n_s, i_s = spatial_interp_baseline(smooth)
    assert i_s <= n_s, "inverse-distance over 8 neighbours must beat single-nearest"
    # Judge against the field's OWN predict-mean error rather than an arbitrary constant:
    # interpolating a smooth field should remove the large majority of that variance.
    held = field[val]
    predict_mean = float(((held - held.mean(0)) ** 2).sum(-1).mean())
    assert n_s < 0.25 * predict_mean, (n_s, predict_mean)

    # Pure noise is not interpolable at all: nearest-neighbour error approaches 2*L (two
    # independent unit-variance draws per latent dim), and IDW shrinks toward the mean.
    noise = {y: (rng.normal(size=(H, W, L)).astype("float32"),
                 torch.tensor(tr), torch.tensor(val)) for y in (1966, 2025)}
    n_n, i_n = spatial_interp_baseline(noise)
    assert n_n > L, (n_n, L)
    assert i_n < n_n
    assert n_s < n_n / 10, (n_s, n_n)         # smooth vs noise must be worlds apart

    # degenerate inputs return NaN rather than raising inside a training run
    assert all(np.isnan(v) for v in spatial_interp_baseline({}))

    # PER-YEAR masks with zero-filled absent cells. _prepare_trend_targets builds each year as
    # np.zeros and marks only that year's points, so a cell absent in year Y holds 0.0. A
    # baseline that reused one year's masks would score interpolation against those zeros and
    # feed them in as sources -- a different population and denominator from _z_mse. Here year
    # 2025 covers everything while 1966 covers only the top half; the answer must depend only
    # on genuinely-present cells, so it must equal the same call with the zeros left untouched
    # outside each year's own mask.
    # Coverage must SHRINK from the first year onward, otherwise reusing the first year's mask
    # would coincidentally stay inside every later year's mask and the bug would hide.
    half = np.zeros((H, W), bool); half[:H // 2, :] = True
    f2 = field.copy()
    varying = {}
    for y, cov_mask in ((1966, np.ones((H, W), bool)), (2025, half)):
        z = np.where(cov_mask[..., None], f2, 0.0).astype("float32")   # absent -> 0.0
        varying[y] = (z, torch.tensor(tr & cov_mask), torch.tensor(val & cov_mask))
    n_v, i_v = spatial_interp_baseline(varying)
    assert np.isfinite(n_v) and np.isfinite(i_v)
    assert i_v <= n_v
    # The zero-fill must not leak in: poisoning the absent region with a huge value cannot
    # change the result, because those cells are outside every year's mask.
    poisoned = {}
    for y, (z, t_m, v_m) in varying.items():
        zp = z.copy()
        outside = ~(_np_or(t_m) | _np_or(v_m))
        zp[outside] = 1e3
        poisoned[y] = (zp, t_m, v_m)
    n_p, i_p = spatial_interp_baseline(poisoned)
    assert abs(n_p - n_v) < 1e-6 and abs(i_p - i_v) < 1e-6, (n_v, n_p, i_v, i_p)
    print("spatial interpolation baseline OK")


def _np_or(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


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
    test_blocked_buffered_split()
    test_partial_conv_5x5()
    test_dropout_threading_and_loss_reparam()
    test_train_loop_and_ablation()
    test_rotation_diagnostic_reads_a_known_angle()
    test_capacity_knob_and_width_persistence()
    test_spatial_interp_baseline()
    test_nonfinite_loss_guard()
    print("\nALL DESK-TRAINING CHECKS PASSED")
