# Initial local Z calibration

Status clarification,2026-10-26: this is a completed modern breeding-only pilot,
not a completed repair of Z. Historical winter and direction-dependent
calibration remain unimplemented. No candidate transform was applied to the
population-model inputs. See
[the current synthesis](DESK_AND_DOWNSTREAM_SYNTHESIS_2026-10-26.md).

The first seasonal PSD-metric calibration completed locally on2026-10-26.1548 encoder-training cells in332 spatial blocks;152 calibration species and121 validation development species. House Finch, sealed species, heldout cells and buffers are excluded. Epochs1996–2005 and2016–2025 use the existing deterministic ABBA split and log1p of mean raw counts.

The fixed candidate family reweights summer/winter blocks, with spatial prior trace normalized on training folds. One shared coefficient field is retained; no split temporal readout is introduced. Selection on calibration-species spatial CV preferred the summer-only endpoint, and that candidate also lowered both level and change kernel losses on validation species.

| Validation-species loss difference, selected minus identity | Point | Conditional95% block-bootstrap interval |
|---|---:|---|
| Spatial community level | −0.004760 | [−0.007123,−0.002504] |
|20-year community change | −0.0000672 | [−0.0001113,−0.0000227] |

Lower loss is better. Units are normalized kernel prediction loss, not correlation or abundance accuracy. The paired6x6 block bootstrap was added after the initial point results, without changing candidates, folds, selection or metric weights. It holds the selected metric, targets and fold normalizations fixed and does not integrate fitting uncertainty. The initial pre-bootstrap result is retained as FINAL_Z_CALIBRATION_INITIAL_RESULTS.json; complete scores and source identity are in FINAL_Z_CALIBRATION_RESULTS.json.

This says the breeding-data community proxy prefers summer geometry in this limited family. It does not confirm global spatiotemporal misalignment, determine a credible historical temporal amplitude, or justify dropping winter information for a year-round population model. The target is a log-abundance community Gram proxy, not the original Ruzicka similarity target. Development species were available to encoder supervision, so withholding them from metric selection is not withholding them from all representation learning.

No production features or priors changed. The exploratory PSD diagonal and fixed support/folds are stored in the WSL FINAL_Z_CALIBRATION_20261026 run root. Direction-dependent metric families, another blocked temporal contrast, and winter-community development objectives remain necessary before proposing a transferable transformation.
