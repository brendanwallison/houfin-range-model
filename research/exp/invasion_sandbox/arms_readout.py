"""Generic replay readout for JSON arms (F3 zero share and per-region abundance fit; review item D9).

For each arm in --spec (plus the fit): per region x era BBS log-lik, observed vs predicted zero share (P(count = 0)
from the fit's own observation model), front offsets and never-reached share. Arms may use any harness key, incl.
k_real, habitat_keys, fmax_scale, allee_gamma_scale.
Usage (inside sbx_env.sh): python arms_readout.py --spec arms.json --out FILE.json
"""
import argparse, json, os
from pathlib import Path
from harness import Sandbox


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sb = Sandbox(os.environ['SBX_REFINED'])
    arms = [dict(name='fitted')] + json.loads(a.spec.read_text())['arms']
    res = {}
    for arm in arms:
        r = sb.run(arm)
        res[arm['name']] = dict(abundance=r['abundance'], offsets=r['curves']['rel_10pct']['arrival_offset']['by_cohort'],
                                never=r['curves']['rel_10pct']['arrival_offset']['never'])
        print(arm['name'], json.dumps({k: (round(v['zero_share_obs'], 3), round(v['zero_share_pred'], 3))
                                       for k, v in r['abundance'].items() if v.get('zero_share_obs') is not None}), flush=True)
    a.out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
