# Competing architecture: mean similarity map plus covariate-driven deviations

This is an additional arm, not a replacement for direct change supervision or
the ongoing time-conditioned DESK pilot. Repository/history searches found the
E073 PartD design, but no verified completed neural DESK run with that design.
The E073 community delta/readout tests and the new CPU anchored-change control
are related implemented comparisons, not a completed mean/deviation DESK.

## Definition

For a declared reference period R, let cbar(s) be the cell's mean environmental
covariates and dc(s,t)=c(s,t)-cbar(s). Fit

    Z(s,t) = Zmean_theta(cbar(s)) + d_phi(dc(s,t); cbar(s)).

The mean map retains DESK's nonlinear stream encoders and spatial neighborhood
branch. A first constrained change head uses

    d_phi(dc; cbar) = U [V dc * gate_phi(cbar)].

Thus reference covariate levels determine how the model responds to covariate
changes. Both levels and deltas enter the complete representation. Static
covariates can modulate the response even though their own deltas are zero.
The first change head is linear in deltas and nonlinear in environmental
context; a fully nonlinear delta head is a later ablation requiring explicit
reference centering. No unrestricted calendar/coordinate input is added here.

With a fixed reference context the linear-delta head has exact zero mean over
the reference years. Its output is exactly zero at zero covariate delta.
Zero initial change-output weights reproduce the mean-only map. This fixes the
mean/deviation offset ambiguity. Rank is a declared community-capacity choice,
not a House-Finch-fitted temporal coefficient or a0.14 scaling.

## Training/export requirements

Use the same physically masked community sources as the other arms. First fit
the mean similarity map on permitted reference-period community averages, then
fit deviations on repeated-observation change vectors while retaining the mean
loss; joint fine-tuning follows. Use the same change objective in the regular
encoder and this architecture so an architecture gain is not confused with a
loss-function gain. Phase1 labels and normalization must exclude all outer
evaluation observations. Reference covariates are predictors, not bird labels.

Reference tensors and reusable mean states are explicit model arguments. They
must be passed through checkpointed calls, saved in metadata and reconstructed
identically at export; a mutable per-year global reference is unacceptable.
With dropout, compute the mean state once per training window and reuse it.

Keep the existing observation contract: training can apply its learned causal
EMA to raw Z; the population cube exports instantaneous raw Z. A mean defined
for the raw latent field need not equal the finite-window mean of the smoothed
observation prediction. Do not silently recenter one and label it the other.
`center_deviations` operates on an explicitly supplied output series and does
not authorize exporting an EMA-smoothed population basis.

The summer/private winter components are summed into one fixed128-coordinate
community representation before focal inference. House Finch uses the same
coefficients for spatial and temporal change and does not estimate this split.

## Implemented status

- `research/exp/z_mean_delta_models.py`: research model and explicit centering
  helper implemented.
- `tests/test_z_mean_delta_models.py`: zero-change anchor, symmetric delta mean,
  primary/private outputs, finite gradients and finite-window centering checks
  pass.
- Full trainer/exporter integration, phase1/phase2 data configuration, full
  training smoke and a queued ecological fit remain outstanding. No trained
  mean/deviation neural checkpoint is claimed.

The existing time-conditioned GPU task continues. The current CPU pilot has
finished fitting/evaluation and does not displace either neural approach.
