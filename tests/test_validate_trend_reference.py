"""The five-comparison panel that grades the encoder against the trend products.

Two kinds of test here. Most check one comparison against a constructed ground truth. The
important ones are the harness checks at the bottom: feed the panel the reference AS the model
and every comparison must return its perfect score; feed it a temporally frozen model and the
temporal comparisons must fall to their nulls. Without those, a panel that silently computed
the wrong thing would still produce plausible-looking numbers.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.validate_trend_reference import (
    cell_epoch_rows, decode_species, fit_species_decoder, similarity_agreement,
    species_sign_agreement, temporal_direction, temporal_rank,
)


def _keys(rows):
    return np.array(rows, dtype="int32")


# ----------------------------- pairing cells to epochs -----------------------------

def test_cell_epoch_rows_picks_the_widest_span_per_cell():
    keys = _keys([[0, 0, 1970], [0, 0, 1990], [0, 0, 2020], [1, 1, 1980], [1, 1, 2010]])
    ei, li, cells = cell_epoch_rows(keys, min_gap=5)
    assert cells == [(0, 0), (1, 1)]
    assert keys[ei[0]][2] == 1970 and keys[li[0]][2] == 2020      # widest, not adjacent
    assert keys[ei[1]][2] == 1980 and keys[li[1]][2] == 2010


def test_cell_epoch_rows_drops_cells_with_too_short_a_span():
    """A cell observed over 2 years has almost no change to measure, so including it would
    dilute the temporal statistics with rows carrying no signal either way."""
    keys = _keys([[0, 0, 2000], [0, 0, 2002], [1, 1, 1970], [1, 1, 2020]])
    ei, li, cells = cell_epoch_rows(keys, min_gap=5)
    assert cells == [(1, 1)] and len(ei) == 1


def test_cell_epoch_rows_handles_a_single_year_cell():
    ei, li, cells = cell_epoch_rows(_keys([[3, 3, 1999]]), min_gap=5)
    assert cells == [] and len(ei) == 0


# ----------------------------- 1: temporal direction -----------------------------

def test_direction_is_one_when_the_model_matches_the_reference():
    rng = np.random.default_rng(0)
    Zr = rng.normal(size=(40, 6))
    Zl = Zr + rng.normal(size=(40, 6))
    Z = np.vstack([Zr, Zl])
    ei, li = np.arange(40), np.arange(40, 80)
    out = temporal_direction(Z, Z, ei, li, np.random.default_rng(1))
    assert abs(out["median_dir_cos"] - 1.0) < 1e-9
    assert out["margin_over_null"] > 0.5


def test_direction_is_minus_one_when_the_model_moves_the_opposite_way():
    rng = np.random.default_rng(0)
    early = rng.normal(size=(30, 5)); late = early + rng.normal(size=(30, 5))
    Zm = np.vstack([early, late])
    Zr = np.vstack([late, early])                 # reference change is exactly reversed
    out = temporal_direction(Zm, Zr, np.arange(30), np.arange(30, 60),
                             np.random.default_rng(2))
    assert abs(out["median_dir_cos"] + 1.0) < 1e-9


def test_direction_null_catches_a_model_that_moves_everywhere_the_same_way():
    """Community change is spatially broad, so a model predicting ONE continent-wide direction
    scores well against almost any reference. The shuffled null is what exposes it: shuffling
    cells does not lower the score, so the margin collapses."""
    rng = np.random.default_rng(3)
    n, L = 60, 4
    early = rng.normal(size=(n, L))
    common = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    Zm = np.vstack([early, early + common])       # same direction at every cell
    Zr = np.vstack([early, early + common + 0.05 * rng.normal(size=(n, L))])
    out = temporal_direction(Zm, Zr, np.arange(n), np.arange(n, 2 * n),
                             np.random.default_rng(4))
    assert out["median_dir_cos"] > 0.9, "raw agreement looks excellent"
    assert out["margin_over_null"] < 0.1, "but the null is just as good, which is the point"


# ----------------------------- 2: temporal rank -----------------------------

def _rotated(angles, seed=5):
    """Cells whose change is an exact rotation by ``angles``, so 1-cos is exactly 1-cos(angle).

    Built from an orthonormal pair rather than by scaling a random step: 1-cos depends on the
    ANGLE between the two vectors, so scaling a step changes the angle differently for cells
    with different base magnitudes and is not a monotone relabelling at all.
    """
    rng = np.random.default_rng(seed)
    n = len(angles)
    base = rng.normal(size=(n, 4))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    perp = rng.normal(size=(n, 4))
    perp -= (np.sum(perp * base, axis=1, keepdims=True)) * base
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    late = np.cos(angles)[:, None] * base + np.sin(angles)[:, None] * perp
    return np.vstack([base, late])


def test_rank_is_one_for_a_monotone_relabelling_of_change():
    """Rank only, so a model whose change magnitudes are a monotone transform of the
    reference's -- which is what the reference's caps do to magnitude -- must still score 1."""
    n = 50
    ang = np.linspace(0.05, 1.2, n)
    Zr = _rotated(ang)
    Zm = _rotated(0.3 * ang)                      # smaller angles, identical ORDERING
    out = temporal_rank(Zm, Zr, np.arange(n), np.arange(n, 2 * n))
    assert out["spearman"] > 0.999, out["spearman"]


def test_rank_falls_apart_when_the_ordering_is_scrambled():
    """The negative control: same magnitudes, shuffled across cells, must not score."""
    n = 50
    ang = np.linspace(0.05, 1.2, n)
    Zr = _rotated(ang)
    Zm = _rotated(np.random.default_rng(0).permutation(ang))
    out = temporal_rank(Zm, Zr, np.arange(n), np.arange(n, 2 * n))
    assert abs(out["spearman"]) < 0.4, out["spearman"]


def test_rank_returns_none_rather_than_crashing_on_too_few_cells():
    Z = np.ones((2, 3))
    out = temporal_rank(Z, Z, np.array([0]), np.array([1]))
    assert out["spearman"] is None and out["n_cells"] <= 1


# ----------------------------- 3: species sign -----------------------------

def test_species_sign_agreement_counts_matching_directions():
    dX = np.array([[1.0, -1.0], [2.0, -3.0]])
    ppy = np.array([[5.0, -2.0], [1.0, 4.0]])     # 3 of 4 signs agree
    out = species_sign_agreement(dX, ppy)
    assert abs(out["frac_same_sign"] - 0.75) < 1e-9 and out["n"] == 4
    assert out["null"] == 0.5


def test_species_sign_skips_near_zero_published_rates():
    """Where the published rate is ~0 its sign is a coin flip in the product itself, so
    counting those entries drags the statistic toward 0.5 whatever the model does."""
    dX = np.array([[1.0, 1.0, 1.0]])
    ppy = np.array([[5.0, 0.001, -4.0]])
    out = species_sign_agreement(dX, ppy, min_abs_ppy=0.5)
    assert out["n"] == 2                          # the 0.001 entry is excluded
    assert abs(out["frac_same_sign"] - 0.5) < 1e-9


def test_species_sign_can_exclude_species_the_decoder_cannot_reconstruct():
    """For a species the decoder fits poorly, comparison 3 measures the decoder rather than
    the encoder, so it must be possible to drop it."""
    dX = np.array([[1.0, 1.0], [1.0, 1.0]])
    ppy = np.array([[1.0, -1.0], [1.0, -1.0]])    # species 1 always disagrees
    r2 = np.array([0.8, 0.01])
    out = species_sign_agreement(dX, ppy, decoder_r2=r2, min_r2=0.2)
    assert out["n_species_used"] == 1 and abs(out["frac_same_sign"] - 1.0) < 1e-9


def test_species_sign_handles_nan_and_an_empty_result():
    out = species_sign_agreement(np.array([[np.nan]]), np.array([[1.0]]))
    assert out["frac_same_sign"] is None and out["n"] == 0


# ----------------------------- the decoder -----------------------------

def test_decoder_recovers_an_exactly_linear_relation():
    rng = np.random.default_rng(6)
    Z = rng.normal(size=(300, 5))
    W_true = rng.normal(size=(5, 3))
    X = Z @ W_true + 2.0
    W, r2 = fit_species_decoder(Z, X, ridge=1e-8)
    assert np.allclose(decode_species(Z, W), X, atol=1e-4)
    assert (r2 > 0.999).all()


def test_decoder_r2_reports_a_species_it_cannot_reconstruct():
    """The reported r2 is what makes comparison 3 interpretable -- a species the decoder
    cannot reach must show a low value rather than silently producing noise."""
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(400, 4))
    good = Z @ rng.normal(size=(4, 1))
    noise = rng.normal(size=(400, 1))             # unrelated to Z
    W, r2 = fit_species_decoder(Z, np.hstack([good, noise]), ridge=1e-6)
    assert r2[0] > 0.99 and r2[1] < 0.2


# ----------------------------- 4 and 5: similarity -----------------------------

def test_similarity_skill_is_one_for_a_perfect_model_and_zero_for_the_null_itself():
    """The harness check for comparisons 4 and 5: a model equal to the null must score exactly
    0 skill. validate_bbs_routes relies on the same identity for its spatial_modern type."""
    rng = np.random.default_rng(8)
    X = np.abs(rng.lognormal(size=(30, 8)))
    Z = rng.normal(size=(30, 5))
    same = similarity_agreement(X, Z, Z_null=Z)
    assert abs(same["rmse_skill"]) < 1e-12
    assert abs(same["cka_gain"]) < 1e-12 and abs(same["mantel_gain"]) < 1e-12


def test_similarity_reports_the_references_own_similarity_level():
    """A reference whose off-diagonal similarity sits near 1 is underpowered -- everything
    resembles everything -- and the panel has to surface that rather than grading against it
    silently. This is the failure mode the reconstructed target sits near."""
    flat = np.ones((20, 6)) + 1e-9
    out = similarity_agreement(flat, np.random.default_rng(9).normal(size=(20, 4)))
    assert out["median_offdiag_reference"] > 0.99


# ----------------------------- harness: a perfect and a frozen model -----------------------

def _panel(Zm, Zr, X_ref, keys, seed=0):
    ei, li, cells = cell_epoch_rows(keys, min_gap=5)
    rng = np.random.default_rng(seed)
    return {"direction": temporal_direction(Zm, Zr, ei, li, rng),
            "rank": temporal_rank(Zm, Zr, ei, li, cells),
            "similarity": similarity_agreement(X_ref, Zm, Z_null=Zm)}


def test_the_reference_graded_against_itself_scores_perfectly():
    """The whole-panel harness check. If any comparison is computing something other than what
    its name says, this is where it shows -- a field that does not reach its perfect value."""
    rng = np.random.default_rng(10)
    n = 40
    keys = _keys([[i, 0, 1970] for i in range(n)] + [[i, 0, 2020] for i in range(n)])
    Z = rng.normal(size=(2 * n, 6))
    X = np.abs(rng.lognormal(size=(2 * n, 8)))
    out = _panel(Z, Z, X, keys)
    assert abs(out["direction"]["median_dir_cos"] - 1.0) < 1e-9
    assert out["rank"]["spearman"] > 0.999
    assert abs(out["similarity"]["rmse_skill"]) < 1e-12


def test_a_temporally_frozen_model_collapses_the_temporal_comparisons():
    """A model that predicts no change at all must score at the null on direction and produce
    no rank agreement -- not a small positive number that could be mistaken for weak skill."""
    rng = np.random.default_rng(11)
    n = 50
    keys = _keys([[i, 0, 1970] for i in range(n)] + [[i, 0, 2020] for i in range(n)])
    base = rng.normal(size=(n, 5))
    Zm = np.vstack([base, base])                              # frozen: zero change
    Zr = np.vstack([base, base + rng.normal(size=(n, 5))])
    ei, li, cells = cell_epoch_rows(keys, min_gap=5)
    d = temporal_direction(Zm, Zr, ei, li, np.random.default_rng(12))
    r = temporal_rank(Zm, Zr, ei, li, cells)
    assert not np.isfinite(d["median_dir_cos"]), "a zero change vector has no direction"
    assert r["spearman"] is None or abs(r["spearman"]) < 0.3
