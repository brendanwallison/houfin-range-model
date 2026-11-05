# What E086's change reliability measures

Recomputed locally from the archived desk_tempho_1995_s1 and desk_tempho_1985_s1 esk_epochs.npz arrays on2026-10-26. No new sealed-species observations were read. This is a point-estimate reproduction, not an independent ecological validation or confidence-interval analysis.

e086_predictability.py computes, separately for each ESK coordinate, correlation across heldout cells between the two observed epoch-change estimates. It then reports Spearman–Brown R=2r/(1+r), interpreting disjoint halves as parallel measurements of a common quantity. It is not per-cell angular agreement. R=.8–.9 corresponds to raw half correlation r=.667–.818. Interpretation as reliability of the full estimate requires assumptions about comparable halves and uncorrelated measurement errors.

| Cache | Cells | Raw half-r mean dims1–24 /25–64 | SB reliability mean dims1–24 /25–64 | Per-cell change-vector cosine median [quartiles] |
|---|---:|---|---|---|
|tempho1995_s1|306|.801/.736|.889/.847|.820 [.718,.902]|
|tempho1985_s1|304|.806/.739|.892/.849|.819 [.721,.906]|

Cosines are computed across all64 projected ESK coordinates at each cell, without spatial centering. The median corresponds to about35degrees. It is substantial repeatability of projected change directions, not nearly perfect alignment. The per-coordinate correlation statistic instead centers across cells. These two summaries answer different questions.

I reconstructed the count halves using the current split_half_groups and epoch_mean_observed functions. No A/B row overlaps occur. Stored x_a/x_b agree with the reconstruction to maximum absolute difference4.77e-7, consistent with stored float32 rounding. Thus accidental reuse of the same survey rows is not indicated by this check.

What this does not establish: independent observers/routes between halves; no temporally persistent observation bias; equal precision or fully independent errors; accuracy in the complete unprojected community; habitat/fundamental-niche change; reliability of1902–1940 winter or absent summer data. Adjacent/interleaved years can share route placement, observers and systematic epoch differences. Shared observation effects can be reproducible in both halves.

Follow-up did occur. E086k rebuilt coordinates to rank128 and reported similar high reliability using odd/even surveyed-year halves and an annual-projection averaging route. E086e used a cross-half observed-change covariance to learn change directions; E073b's skeptic checks also estimated high reliability for a different development-species target. These are related repeatability checks, not independent observer/route validation of the original claim. E086k explicitly cautioned that persistent route effects count as reliable. The existing E028/E028b observer analyses show substantial observer-associated temporal structure, but do not directly re-estimate the original E086 statistic after removing observer effects.

Corrected interpretation: DESK missed much of the repeatable change in the projected observed community for these older BBS targets. These results alone do not quantify how much of that gap is ecological information available to a habitat model. The previous shorthand that the falls are 'DESK's, not survey noise' is too strong if read as excluding systematic observation effects.

Sources: research/exp/e086_predictability.py; research/exp/esk_project.py; validation_core.split_half_groups/abba_halves; validate_bbs_routes.epoch_mean_observed; E086.md initial predictability and E086k sections; E028.md/E028b; research/skeptic/E073b/check.py. Reproduction launcher: .reliability_audit.sh (temporary workspace helper).
