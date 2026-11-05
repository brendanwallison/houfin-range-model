"""Review 2 item 4a: decompose R40 / R42-R40 log G(t, cell) = g0 + Zstd . (b_scale b) into a continental yearly mean
(uniform clock), a regional (East / Plains / West yearly mean minus continental) and a local residual; per-dim
contributions to the East 1940 -> 2020 rise."""
import os, json, sys, numpy as np
from harness import Sandbox
sb = Sandbox(os.environ['SBX_REFINED'])
Z = np.asarray(sb.data['Z_gathered'], float)
zmu, zsd = Z.reshape(-1, Z.shape[-1]).mean(0), Z.reshape(-1, Z.shape[-1]).std(0) + 1e-9
Zs = (Z - zmu) / zsd; Mz = Z.shape[-1]
out = {}
for tag, path in (('R40', sys.argv[1]), ('R42_R40_zdisp_real', sys.argv[2]), ('A9', sys.argv[3])):
    e = np.load(path)['estab']; b = .1 * e[1:1 + Mz]
    lin = e[0] + Zs @ b                                         # (T, N)
    band = sb.region['east'] | sb.region['plains'] | sb.region['west']
    cont = lin[:, band].mean(1)                                 # continental clock
    r = {}
    for yr0, yr1 in ((1940, 2020), (1966, 2020)):
        i0, i1 = int(np.nonzero(sb.years == yr0)[0][0]), int(np.nonzero(sb.years == yr1)[0][0])
        d_cont = cont[i1] - cont[i0]
        row = {'continental_clock_dlogG': float(d_cont)}
        for reg in ('east', 'plains', 'west'):
            m = sb.region[reg]
            d_reg = lin[i1, m].mean() - lin[i0, m].mean()
            dloc = (lin[i1, m] - lin[i0, m]) - d_reg
            row[reg] = dict(total_dlogG=float(d_reg), regional_minus_clock=float(d_reg - d_cont), local_sd=float(dloc.std()))
        m = sb.region['east']
        contrib = (Zs[i1][:, :][m].mean(0) - Zs[i0][m].mean(0)) * b
        top = np.argsort(-np.abs(contrib))[:6]
        row['east_top_dims'] = {int(k): float(contrib[k]) for k in top}
        row['east_top6_share'] = float(np.abs(contrib[top]).sum() / np.abs(contrib).sum())
        r[f'{yr0}-{yr1}'] = row
    out[tag] = r
print(json.dumps(out, indent=1))
