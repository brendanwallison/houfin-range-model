# Downstream contract (frozen 2026-10-08 at research/desk-temporal 26fd366)

The loop anchors every "better" on THIS contract. The age model keeps evolving (run_19 -> run_20 on main);
changes there are tracked, not chased, and this file is updated only at a phase gate with the user.

## What the age model consumes
- **Features:** the cube's RAW instantaneous DESK z (`build_final_z_cube.py`; `temporal_output ==
  raw_instantaneous` is enforced in `src/data/combine/model_inputs.py:261`), years 1902-2025, truncated BY
  POSITION to the first 24 of 64 dims (`model_inputs.py:470`, `config/age_model_config.json:latent_dim`).
- **Fields:** H_j = z_t . beta_j for adult survival, reproduction, capacity, juvenile survival
  (`src/model/age_fields.py:176-202`); intercepts alpha_j separate; capacity also gets a continental time
  basis `k_trend_basis . w_k_trend` with a near-zero prior ("a safety valve, not a mechanism",
  `age_fields.py:216`); disease depresses K from 1993; Z_disp = path-integrated z feeds journey survival.
- **Prior:** beta_j iid ACROSS FEATURES given an amplitude w_scale_j (rank-2 or exchangeable coupling across
  the four fields; `docs/PRIORS.md` "Manifold prior", "GP amplitude"), so each H_j is a GP with kernel
  w_scale_j^2 z(x,t).z(x',t'). Each eigen-direction of z gets prior variance proportional to its eigenvalue.
- **Data in 1902-1939:** only pre-invasion pseudo-zeros: non-arrival absences more than 700 km beyond the
  native hull, subsampled at the real-data density (`model_inputs.py:199-363`). The release is 1940.

## What a proposed change must keep
- BLR form: features -> linear -> GP with an explicit kernel; a change may add BLOCKS with their own iid
  amplitude (e.g. s_b^2 zbar.zbar' + s_w^2 dz.dz', a time basis, a persistent spatial residual), each a
  kernel the age model can adopt by giving that block its own w_scale.
- Interpretability: every block names what it represents (place, change, trend, residual).
- No use of House Finch data anywhere in the encoder or its evaluation.

## What "better" means (PROTOCOL.md "Purpose")
Withheld-decade level and change accuracy (Great Plains zone separately), attenuation slope of predicted on
true change, interval coverage / implied temporal kernel, responsiveness to absence-only information -- then
one sealed-set confirmation and the human-gated MAP check before anything changes in production.
