"""Difference two row_ll_dump files: where does fit B beat fit A? Sums (ll_B - ll_A) over TRAIN rows by region x era and
by count bin, and route-effect penalty differences by region. Positive = B better.
Usage: row_ll_compare.py A.npz B.npz [labelA labelB]"""
import sys, numpy as np
HELD = '--held' in sys.argv; sys.argv = [x for x in sys.argv if x != '--held']
A, B = np.load(sys.argv[1]), np.load(sys.argv[2]); la, lb = (sys.argv[3:5] if len(sys.argv) > 4 else ('A', 'B'))
tr = (A['train'] & B['train']) if not HELD else (~A['train'] & ~B['train'] & np.isfinite(A['ll']) & np.isfinite(B['ll']) & (A['idx'] >= 0)); d = np.where(tr, B['ll'] - A['ll'], 0.)
lon, yr, y = A['lon'], A['year'], A['y']
regs = dict(west=lon < -100, plains=(lon >= -100) & (lon < -95), east=lon >= -95)
eras = [(1966, 1979), (1980, 1993), (1994, 2005), (2006, 2014), (2015, 2020), (2021, 2025)]
print(f'll({lb}) - ll({la}) over train rows (positive = {lb} better): total {d.sum():.1f}')
print('region   ' + ' '.join(f'{e0}-{str(e1)[2:]:>3s}' for e0, e1 in eras) + '   total')
for r, m in regs.items():
    print(f'{r:8s} ' + ' '.join(f'{d[m & (yr >= e0) & (yr <= e1)].sum():8.1f}' for e0, e1 in eras) + f' {d[m].sum():8.1f}')
bins = [(0, 0), (1, 2), (3, 9), (10, 29), (30, 99), (100, 10 ** 6)]
print('count bin: ' + '  '.join(f'{b0}-{b1 if b1 < 10 ** 6 else "+"}: {d[(y >= b0) & (y <= b1)].sum():.1f}' for b0, b1 in bins))
# East by count bin x era
for e0, e1 in eras:
    m = regs['east'] & (yr >= e0) & (yr <= e1)
    print(f'east {e0}-{e1} by bin: ' + '  '.join(f'{d[m & (y >= b0) & (y <= b1)].sum():.1f}' for b0, b1 in bins))
# mean log mu ratio by region x era (does B predict more or less?)
lr = np.where(tr, np.log(B['mu']) - np.log(A['mu']), np.nan)
for r, m in regs.items():
    print(f'mean log(mu_{lb}/mu_{la}) {r}: ' + ' '.join(f'{np.nanmean(lr[m & (yr >= e0) & (yr <= e1)]):+.3f}' for e0, e1 in eras))
pe = .5 * (B['route_eps'] ** 2).sum() - .5 * (A['route_eps'] ** 2).sum()
print(f'route-effect penalty difference (B - A, lower = B better): {pe:.1f}')
