# TACC handoff: DESK trained from 1902 with CBC supervision (E086w)

## What to run

Two DESK trainings on TACC, each followed by the cube stage:

| Run | Overlay | Writes to |
|---|---|---|
| CBC build | `config/overlays/desk_cbc1902.json` | `$HOUFIN_PROCESSED/encoder/desk_cbc1902/`, `$HOUFIN_PROCESSED/encoder/cube_desk_cbc1902/` |
| Matched control | `config/overlays/desk_ctl1902.json` | `$HOUFIN_PROCESSED/encoder/desk_ctl1902/`, `$HOUFIN_PROCESSED/encoder/cube_desk_ctl1902/` |

Both overlays redirect `paths.desk_output_dir`, `latent_cube.model_path` and `latent_cube.output_dir`. The production
products in `encoder/desk/` and `encoder/cube/` are not touched. Do not change those three keys back.

What the CBC build changes relative to production:

- **Training window from 1902.** `desk.output_ema.warmup_start` goes from 1940 to 1902, so DESK forwards and supervises
  1902-2025, not 1940-2025.
- **CBC presence term on 1902-1965 lists.** This is the E069b / E078 presence-only term with per-species completeness
  slopes, weight 0.01, on 15,378 resident-species CBC lists. Early CBC enters only as presence with completeness,
  because E070 found count-based similarity does not survive pre-1940 damage.
- **CBC Ruzicka term on 1966-2024 winters.** This is the E081 recipe, weight 2.36 with fixed calibration
  dot = -0.045 + 0.707 x Ruzicka, on 81,795 cell-years.

The control has the same window and no CBC terms. It is needed because the 1902 window by itself changes the output
EMA's burn-in, so CBC effects can only be read against it.

## Code

Check out branch `research/desk-temporal` at the commit that added this file, or later. Never use `main`. The
presence and Ruzicka terms (`desk.aux_presence`, `desk.aux_ruzicka`) exist only on this branch.

## Data to upload

Both files are on the local machine in `E:\Datasets\cbc_raw\prepared\tacc_1902\` (WSL:
`/mnt/e/Datasets/cbc_raw/prepared/tacc_1902/`). Copy them to `$HOUFIN_PROCESSED/encoder/cbc_aux/` on TACC, which is
`$WORK/houfin/processed/encoder/cbc_aux/`.

| File | Size | sha256 |
|---|---|---|
| `cbc_presence_res_comp_1902_1965.npz` | 157 KB | `40579970e82340bba97c045cb6e09f175e098c6e0f36760d9666bd4568831595` |
| `cbc_ruzicka_res.npz` | 4.1 MB | `3cb9a6be5462c04186777154c89cb27b6cfcf71a8ae64701a692de8aac7d3058` |

Verify with `sha256sum` after the copy. Neither file contains House Finch data or any sealed species. Both use only
the 59 resident reference species.

Everything else the run needs (the yearly covariate states from 1902, the ESK basis, BBS trend targets) is already on
TACC from the production pipeline. The states have covered 1902-2025 since the climate pipeline was built
(`config/data_config.json`, first_year 1902).

## Commands

From the repo root on TACC (`scripts/tacc/env.sh` sets `HOUFIN_PROCESSED` and the allocation):

```bash
mkdir -p $WORK/houfin/processed/encoder/cbc_aux        # then copy the two files in, check sha256

# CBC build
ESK_DESK_CONFIG=config/overlays/desk_cbc1902.json STAGES="desk cube" TIME=12:00:00 \
    bash scripts/tacc/submit_encoder.sh

# matched control (can run at the same time on another node)
ESK_DESK_CONFIG=config/overlays/desk_ctl1902.json STAGES="desk cube" TIME=12:00:00 \
    bash scripts/tacc/submit_encoder.sh
```

Before submitting, confirm that `submit_encoder.sh` passes `ESK_DESK_CONFIG` through to the job. It uses
`--export=ALL`, so an exported or inline variable reaches `run_encoder.py`, which reads it in
`src/config_utils.load_config`.

**Resources.** The queue is `gpu-a100-small` (A100 40 GB). A local run with the 1940 window peaks at 15.4 GB allocated.
The 1902 window is 124 years against 86, so expect roughly 22 GB. That fits 40 GB but would not fit a 24 GB card.
The year window is staged through host memory, so if the job dies with a HOST (not CUDA) out-of-memory error,
resubmit with `QUEUE=gpu-a100`. Wall time should be about 1.5x a normal production DESK run; 12 h is a generous cap.

## Checks in the log (`houfin_encoder.o<jobid>`)

Early in the run:

- `[desk] aux_presence: N of 15,378 lists kept (window 1902-2025, ...)`. The window must start at **1902**; if it
  says 1940 the overlay was not applied. N will be lower than 15,378, because held-out, buffer and invalid cells are
  removed (expect very roughly 10-13k).
- `[desk] PRESENCE term 'cbc_presence_early': ... lists, 59 species ...`
- `[desk] aux_ruzicka: ... of 81,795 CBC cell-years kept (withheld years INCLUDED)`
- `[desk] RUZICKA term 'cbc_ruzicka_modern': ... dot ~ -0.045 + 0.707 x Ruzicka, 4096 pairs/step, weight 2.36`
- The control's log must show **neither** term, and the same 1902 window.

Every evaluated epoch:

- `Ep NNN | ruzicka 'cbc_ruzicka_modern' mse ...` and a presence-loss line, each a few percent of the total loss.
- A `VRAM a/b` field. The peak should be about 22 GB.

## What to bring back

For each run, from `$HOUFIN_PROCESSED/encoder/desk_{cbc,ctl}1902/`:

- `run_summary.json`, `desk_meta.npz`, the training log (`houfin_encoder.o<jobid>`), the GPU monitor CSV
  (`gpu_encoder_<jobid>.csv`);
- `env_model_semisup.pth`, the selected checkpoint;
- the latest-epoch checkpoint, if the run keeps one.

The cube directories (`cube_desk_{cbc,ctl}1902/`, `Z_latent_<year>.npy` for 1902-2025) are needed for the downstream
age model. They are large, so copy them only if the downstream run happens locally.

## Why, and how much is established

- **CBC Ruzicka supervision is established for sedentary species.** E086s / E086u found it raises sparse-data change
  skill: +0.06 at 3% target data at the selected state, 4 / 4 seeds, no CI below 0, plus +0.05 in the level readout.
  It also repairs the pre-1940 winter backcast (E086p). The sparse-data gain needs CBC winters inside the target era
  (E086u: 21% of it survives without them), which is why the early-era lists are in training at all.
- **Early lists as a presence term with completeness slopes are NOT yet established.** E086v (running locally) tests
  whether 1966-95 winters degraded to pre-1940 quality still carry the gain, through this same presence route. Until
  it reports, the 1902-1965 presence term in this build is a reasoned choice: it is the only route E070 found to
  survive early damage.
- **Untested here:** the 1902 window's own effect on z (hence the control), and breeding-season communities before
  1966 (no data exist).

Full records: `research/experiments/E086.md`, entries E086n-E086v.
