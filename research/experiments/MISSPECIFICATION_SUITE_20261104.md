# Abundance, shooting and establishment discrepancy suite — November 4, 2026

Implementation and execution record: 18 designs × seeds 41/42, staged as isolated
`step48_*` TACC experiments. Publication and live execution receipts are recorded
separately; staged is not synonymous with queued, running, or fully validated.
The local GPU remains reserved for Claude.

Verified publication at **03:36 CST, November 4**: all 36 task files are pending,
and the rolling dispatcher lists the new suite first after any administrative
handover tasks. Its first three scientific claims will be `ab_ar15_weak_s41`,
`est_static10_full_s41`, and `shoot10_weak_s41`. The resource watcher was refreshed
and its live snapshot confirms all 36 are tracked. No new Slurm allocation was
requested for this publication.

Current allocations remain pending: dev job 3550613 estimates 05:40 CST, small
3548970 estimates 11:40 CST; regular jobs 3547292/3548331 have no estimates.
These times are scheduler predictions, not reservations.

Receipts: [queue publication](MISSPECIFICATION_QUEUE_RECEIPT_20261104.json),
[small-grid validation](misspecification_20261104_VALIDATED.json), and
[all design specifications](misspecification_20261104_specs.json).
Full-data GPU smokes have **not** run; they are mandatory task gates.

## Literature and project evidence

Weak-constraint data assimilation explicitly estimates departures from a physical
transition model, with the error covariance specifying which spatial and temporal
departures are plausible. Trémolet also demonstrates that estimated model error
can absorb observation bias; a better assimilated trajectory alone does not prove
a better forecast model. We therefore retain observation-only/unaltered controls,
raw likelihood scoring, held-out records, and discrepancy-free replays.

- [Trémolet, Model error estimation in 4D-Var (2007)](https://www.ecmwf.int/sites/default/files/elibrary/2007/12819-model-error-estimation-4d-var.pdf).
- [Laloyaux and Bonavita, ECMWF weak-constraint model-bias treatment (2020)](https://www.ecmwf.int/en/newsletter/163/meteorology/improving-handling-model-bias-data-assimilation).
- [Farchi et al., Using machine learning to correct model error in data assimilation and forecast applications (2021)](https://arxiv.org/abs/2010.12605).
- [Iakovlev et al., Latent Neural ODEs with Sparse Bayesian Multiple Shooting (2023)](https://arxiv.org/html/2210.03466v2).

These support the method families, not ecological correctness of our particular
gates, priors, or scales. We use small explicit discrepancy terms rather than a
neural-network emulator. The shooting model learns segment initial states with
probabilistic continuity; the implementation remains a sequential JAX scan and
does not claim parallel integration in time.

Claude's `invasion_sandbox` R22–R25/R43 work is an important design input:
abundance adjustments can greatly improve counts while leaving arrival timing
poor; persistent place-specific establishment adjustments can move timing much
more efficiently. Some sandbox arms froze habitat/readouts, used imperfect
low-density gates, or stopped at iteration caps. They motivate full joint-fit
tests here, rather than serving as established results on this base. Our gate
uses the actual absolute/clumped mate-finding law, not N/K. Earlier step11 had a
single 1966 multiplicative reset and barely used it; that does not test independent
shooting states that cut long trajectory derivatives.

## Common base and temporal growth controls

All arms retain yearling breeding, original Veit departure, the fixed weak
absolute/clumped Allee effect, K floor 4× its pinned scale, nominal established
1951 founders, original pseudozeros, the inherited NestWatch block, and real
time-varying K and Z_disp. No new CBC data likelihood is added. The NestWatch
annual conversion remains fixed; that limitation is inherited, not resolved here.

Three temporal-growth settings retain the spatial growth readouts but blend their
link-scale histories toward their 1966 spatial maps:

- Full: 100% of temporal variation in adult survival, juvenile/adult survival ratio
  and maximum reproduction.
- Weak: 25% of that temporal variation on their respective links.
- Frozen: 1966 spatial growth fields repeated over years.

K and journey habitat are not frozen. Crowding/equilibrium quantities are
recomputed after this restriction. A smaller historical lambda change is partly
imposed in the restricted arms and cannot itself count as an improvement: the
question is whether likelihood, invasion timing and the other pathologies remain
acceptable, and how much correction is needed to achieve that.

## Designs, each with two optimizer seeds

| Family | Designs |
|---|---|
| Controls | Full, weak and frozen temporal growth, no discrepancy |
| Established-abundance corrections | Eight smooth spatial modes, stationary AR(1) rho .9, annual log scales .05/.15 under weak growth; .15 under full/frozen growth |
| More local abundance corrections | Independent per-cell five-year blocks, log scale .15, weak growth |
| Persistent establishment | One parameter per cell, log scales .5/1 under full growth; scale 1 under frozen growth |
| Arrival redistribution | One parameter per cell, log scale 1, full growth; changes settlement allocation, conserving total juvenile arrivals |
| Multiple shooting | Independent adult and young initial states at 10-year boundaries, continuity log SD .15/.5; 5-year boundaries at .15; weak growth |
| Occurrence-focused comparisons | Full-growth control, weak-growth abundance .15, and full-growth establishment 1; conditional positive-count likelihood weighted .25 |

All calendar discrepancy fields are active in 1951–2020 and zero afterward.
Shooting boundaries begin in 1956 and end no later than 2020; none occur in
temporal holdout years after 2020. Spatially held-out cells may have latent
states, but their observations retain zero fitting weight.

## Precise discrepancy semantics

For abundance corrections, let m(N) be the existing absolute mate-finding fraction.
After the biological transition, multiply both ages by
`exp(bounded_field(t, cell) * m(N)^2)`. Thus empty cells remain empty, age shares
are preserved, and the correction vanishes quadratically toward zero density.
This places more flexibility in established abundance than in sparse establishment.
It does not hold realized growth fixed at all positive densities.

Establishment corrections multiply maximum reproduction immediately before local
reproduction by `exp(bounded_field(cell) * (1-m(N)) * exp(-E/10))`.
E is a smooth years-occupied clock, incremented by `N/(N+1/gamma)` after each year.
The same persistent place effect can act through a cell's establishment period,
then fades with occupation and mate availability. Zero corrections reproduce the
base exactly. The clock and ten-year scale are explicit modeling assumptions.

Arrival corrections apply destination weights `exp(bounded_field(cell)*(1-m(N)))`
to juvenile arrivals and normalize to preserve their continental total. No birds
are manufactured and zero arrivals stay zero. This is a destination-allocation
discrepancy, not a fitted survival probability. Global normalization can alter
distant allocation simultaneously; its effective spatial reallocation is audited.

All continuous correction fields use `2*tanh(sigma*u/2)` with standard-Normal
coefficients. The quoted sigma is the small-signal log scale, not fitted RMS.
Median-one centering makes zero innovations exactly the identity; we deliberately
avoid the negative drift of mean-one centering in a joint-mode comparison.

Shooting learns positive adult and juvenile states separately at each boundary.
Their log states have a Normal continuity density centered on the log of the
previous segment endpoint (floor 1e-20 internal units), at fixed SD .15 or .5.
The next segment starts from these independent states, so observation gradients
do not have to pass through the entire preceding biological trajectory. The
continuity penalty still couples segments. A broad Normal proposal centered on
the old trajectory initializes the nodes; its density is subtracted exactly,
leaving only the endpoint continuity model, not an empirical baseline-state prior.
Numerical floor seeding and gross/net boundary additions must be reported.

The occurrence-focused loss is a generalized likelihood sensitivity, not a new
independent data source or occupancy model. For real survey records decompose the
same NB2 log likelihood into occurrence and positive-truncated count components.
Retain the first and weight the latter by .25. Pseudozero terms and inherited
holdouts retain their original treatment. Unweighted NB2 and occurrence scores
are saved for fair comparison. Counts, zeros and presences are not counted twice.

## Validation and interpretation requirements

Small-grid checks precede publication: zero-correction identity, nonzero finite
gradients, shooting endpoint identity, proof that shooting cuts earlier-state
derivatives, bounded corrections, stationary AR recurrence, exact hurdle
decomposition, and code-manifest integrity. Every queued fit additionally must
pass an allocated-GPU full-data joint-likelihood/gradient and numerical accounting
smoke before optimization. Small-grid validation is not full-data validation.

Each accepted fit runs the existing pathology and invasion summaries plus amended
origin-attribution accounting. Corrections and shooting additions are recorded
separately from biological births/deaths. The contribution calculation freezes
ambient fitted gates/settlement weights, as required for linear origin attribution;
it is not a nonlinear isolated-population forecast.

Follow-up outputs use identical East/Plains/West support (25–49°N; boundaries
100°W/95°W), historical and post-1966 growth changes, raw real-record count and
occurrence scores, inherited holdouts, fixed absolute density threshold arrival
maps, full-versus-discrepancy-free trajectories, and correction budgets. Standard
front counterfactuals that retain shooting nodes are conditional on those anchors;
the discrepancy-free replay is essential for interpreting biological spread.

Do not select on total joint objective across different dimensions/loss weights.
Review K/Allee nonviability separately from genuinely unfavorable Plains habitat,
intrinsic and realized source/sink structure and bridges, age/survival tradeoffs,
reproduction, immigration dependence, timing, correction takeover and prediction.
Dynamic-gate isolation, finite-propagule experiments and HMC geometry remain
explicit follow-ups; they are not silently treated as passed.
