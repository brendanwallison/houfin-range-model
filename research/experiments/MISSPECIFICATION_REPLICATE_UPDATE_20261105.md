# Replicated misspecification results — November 5, 2026, afternoon

## Execution

At 14:18 CST: 22 scientific tasks complete, three running, two pending, six
failed, three held (36 total). One capped-but-improving block-correction fit
has since been returned to pending: **22 done, three running, three pending,
five failed, three held**, before any subsequent task completions. Separate
diagnostic-recovery tasks are outside those 36 scientific designs.

By **14:26 CST**, the second regular allocation had also started. Both regular
nodes are active: **22 done, six running, five failed, three held**. All six
remaining eligible scientific tasks are running; none of these 36 is currently
pending. Dev 3557354 remains pending for continued authorized backlog work.

Regular allocation 3547292 started at 11:55 CST and has three active lanes.
Regular 3548331 and dev 3557354 remain pending, with no start estimates. Dev
3556610 completed 08:06–10:04. The watcher is connected and registering progress;
the original continuation ceiling remains unchanged. Local GPU use remains off.

The active scientific tasks at the check were the frozen-growth control,
frozen-growth plus abundance correction, and independent five-year block
abundance corrections, all seed 42. The seed-41 block fit hit 16,000 refinement
iterations while still improving (~.4 objective units over the last 20 steps).
It resumes the same nonterminal checkpoint with a 32,000 limit, not a new MAP.
The watcher now allows the same single bounded extension for the second block
replicate if it reaches the old cap while still improving. Stalled and nonfinite
fits are not automatically retried by that rule.

## What replicates and what does not

All numerical comparisons below use the same definitions and saved-output
analysis as the earlier reports: equal land cells in 25–49N, East/Plains/West
split at 95W/100W; growth change 2010–2025 minus 1902–1915; one fixed internal
arrival threshold .008027860278758616 and 782 eastern cells surveyed before their
first positive record. First detection is not exact biological colonization.

### Arrival redistribution is reproducible, with real costs

| Quantity | Seed 41 | Seed 42 |
|---|---:|---:|
| Historical growth change East / Plains / West | .597 / .435 / .359 | .585 / .433 / .349 |
| Median arrival minus first detection | −1 yr | −1 yr |
| Observed cells not reaching the fixed threshold | 9.7% | 9.5% |
| Plains mean contribution | .737 | .741 |
| Plains contribution sink area | 90.8% | 91.1% |
| Plains local-replacement sink area | 92.9% | 92.9% |
| Plains mean immigrant share | .310 | .307 |
| Plains K/Allee nonviability area | 0% | .26% |
| Median adult stayer survival | .582 | .620 |
| Median juvenile/adult stayer survival ratio | .274 | .250 |
| Mean realized reproduction at nests | 4.43 | 4.32 |

The two no-correction full-growth controls have historical changes
1.246/.733/.817 and 1.456/.886/.924, and median arrival offsets −3 years.
Thus the arrival arm's reduction in historical change and improved median timing
are reproducible under this model setup. The high fraction of genuinely poor
Plains habitat is not itself a pathology. The tiny-K/Allee route is assessed
separately from that legitimate barrier.

Costs remain: both arrival fits have worse raw held-out count scores in all
three regions than their respective controls, higher realized reproduction at
nest locations, and more observed cells below the fixed arrival threshold.
This is a repeatable tradeoff, not clearance of the full pathology checklist or
proof that the destination-reweighting interpretation is biologically correct.

### Establishment effects help, but larger-scale benefits are less stable

The .5 establishment arm is very reproducible: historical changes .982/.600/.679
and .993/.624/.694; adult medians .647/.652 and juvenile/adult ratios .333/.333.
Its cost is increased Plains K/Allee nonviability: 4.47% and 3.95%, versus .79%
and 0% in the corresponding controls. Median timing improves to −2 years but
about 9.1% of observed cells never reach the common threshold.

The scale-1 establishment arm reduces growth changes more strongly, to
.568/.397/.394 and .620/.459/.507. However, adult survival moves .545→.709,
Plains immigrant share .365→.530 and K/Allee nonviability 0→3.68% across seeds.
Its realized contribution mean in the Plains stays near .70, illustrating why
component instability, process-level stability and flux dependence must be
reported separately. It does not reproduce the arrival arm's full combination
of growth, flux and low K/Allee nonviability as consistently.

### Abundance flexibility has not supplied a coherent remedy

- Reduced-growth controls reproduce small historical changes around .31/.22/.24
  with little count-score loss, but adult survival around .36 and juvenile/adult
  ratios .94–.96. Their smaller historical change is imposed by the structure.
- Smaller (.05) abundance corrections give growth changes .351/.230/.253 and
  .256/.235/.259. Yet adult survival/juvenile ratio switch from .373/.934 to
  .587/.360. This is a particularly clear optimizer-mode tradeoff; one coherent
  biological interpretation cannot be inferred from a single seed.
- Larger (.15) abundance corrections under reduced growth give
  .353/.258/.323 and .422/.269/.361, with worse raw held-out count scores in all
  regions than their reduced-growth controls. They are used substantially but
  do not reduce the historical change relative to those controls.
- With full growth flexibility, the completed .15 abundance arm increases
  historical change to 1.399/.982/1.441, especially in the West, while its
  juvenile/adult median ratio reaches .971. Providing that outlet did not force
  the model away from growth inflation.
- Frozen growth fits retain fairly similar count scores, but the no-correction
  control has adult survival .286 and juvenile/adult ratio .948. Frozen growth
  plus abundance correction gives .516/.969; frozen growth plus establishment
  gives .234/.945 and mean nest reproduction 4.94. These imposed-zero-change
  fits expose compensating biological costs; they do not establish that growth
  really was constant.

### Occurrence-focused loss and incomplete cases

Occurrence-focused control and abundance arms repeat their poor tradeoff across
seeds: substantially earlier fronts and worse raw count prediction, despite
improving their targeted occurrence component. This does not reject historical
range supervision or all detection models; it rejects an easy interpretation of
this particular count downweighting as solving wavefront fit.

Three shooting fits stalled at large gradients, and their second-seed queue
files are held intact pending conditioning work. Both occurrence-focused
establishment fits completed refinement but failed negative-flow QC in source/
sink accounting. A separate allocated-node evidence task is queued to quantify
the negative flows and kernel residuals, retaining the QC gate and recovering
the second fit's summary. Neither is silently accepted as fully audited.

## Interpretation limits

Coast-to-coast connectivity was checked for all 22 completed maps, on the same
geographic support and with both edge-only and diagonal connectivity, at source
thresholds 1 and 1.05. **None has a continuous source path under contribution or
actual-state local replacement.** This is a connectivity statement, not a claim
that no individual cells change source/sink class. Potential-growth maps do
produce edge-connected coast-to-coast paths in the full-growth .15 abundance arm
and all four completed occurrence-focused control/abundance fits, even at 1.05.
Ten potential-growth maps have a diagonal-connected path at threshold one.
These intrinsic-versus-realized differences are retained rather than treating
either classification as the whole biological diagnosis.

Arrival redistribution is the most reproducible new mechanistic comparison so
far. This is not an overall model recommendation: prediction, reproduction,
unreached observations and correction dependence remain costs. The comparisons
are point estimates from two optimizer seeds, not posterior identification.
Dynamic isolation, permanent dispersal-distance checks, finite-propagule tests
and HMC geometry remain outstanding. Annual NestWatch conversion and historical
founder/CBC supervision are unchanged in this suite.

Evidence: refreshed `MISSPECIFICATION_MATCHED_SUMMARY_20261104.json` and
`MISSPECIFICATION_RESULTS_20261104.json`, `MISSPEC_LIVE_LATEST.json`, and
`BLOCK_REFINEMENT_CONTINUATION_20261105.json`.
