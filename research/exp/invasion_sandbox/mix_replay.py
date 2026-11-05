"""Which vital rate's share of Z's time change carries the fit? Mixed real / frozen-habitat replays (no refitting).

The fitted parameters are replayed twice through the model's own capture: with the real Z (the fit) and with Z
frozen at its 1966-2025 mean. Each arm takes the listed rate arrays (Sa, Sj, Fmax, K; c recomputed from the mixed
Sa / Sj / Fmax as harness.c_of) from the REAL-Z capture and the rest from the FROZEN-Z capture, then runs the forward
and scores the full BBS log-likelihood under the fit's observation model (25-49N rows), with arrival offsets and the
never-reached share. Journey survival Q / Z_disp stays frozen in every arm except 'all' (R15: it does not time the
front). Arms: none, all, each rate alone, and each leave-one-out.
Usage (inside sbx_env.sh): python mix_replay.py --out DIR
"""
import argparse, json, os
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from harness import Sandbox, capture, edit_habitat, c_of

RATES = ('Sa', 'Sj', 'Fmax', 'K')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sb = Sandbox(os.environ['SBX_REFINED'])
    ref = sb.run(dict(name='fitted'))
    d_fz = edit_habitat(sb.data, sb.years, 0., [1966, 2025])
    x_real, _ = capture(sb.data, sb.lat)
    x_fz, _ = capture(d_fz, sb.lat)
    arms = [('none', ()), ('all_rates', RATES)] + [(f'{r}_only', (r,)) for r in RATES] + \
           [(f'all_but_{r}', tuple(q for q in RATES if q != r)) for r in RATES]
    rows = []
    for name, real in arms:
        fa = {}
        for r in RATES:
            fa[r + '_flat'] = jnp.asarray((x_real if r in real else x_fz)[r + '_flat'])
        Sa = jnp.minimum(fa['Sa_flat'], .995); Sj = jnp.minimum(fa['Sj_flat'], Sa)
        fa.update(Sa_flat=Sa, Sj_flat=Sj, c_flat=c_of(fa['Fmax_flat'], Sa, Sj, x_real))
        row = sb.run(dict(name=name, habitat_alpha=0, forward_args=fa))
        ab = row['abundance']; reg = lambda r_: -sum(v['loglik'] for k, v in ab.items() if k.startswith(r_))
        o = row['curves']['rel_10pct']['arrival_offset']
        rec = dict(arm=name, real_rates=list(real), nll=-sum(v['loglik'] for v in ab.values()),
                   east=reg('east'), plains=reg('plains'), west=reg('west'), offsets=o['by_cohort'], never=o['never'],
                   east_obs_over_expected=[ab[k]['obs_over_expected'] for k in ab if k.startswith('east')])
        rows.append(rec)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    out = dict(fitted_nll=-sum(v['loglik'] for v in ref['abundance'].values()), arms=rows)
    (a.out / 'mix_replay.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
