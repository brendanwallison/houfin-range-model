"""Build the ESTABLISHMENT-WINDOW constraint for laplace_fit.py --establishment_dates (user, 2026-11-06).
Per state with documented dates (research-agent compilation, quotes in SANDBOX_INVASION_SHAPE_2026-11.md ESTABLISHMENT-LAG
LIT): the window [first occurrence record, first breeding record]. A two-sided interval: established neither before any
bird was recorded nor after breeding was documented. AL's first occurrence is unverified -> upper bound only (occ = 0).
State cells = the 54 km cells holding the state's BBS routes (Routes.csv), as in state_lag.py.
Usage: establishment_build.py --input INPUT_DIR --out NPZ
"""
import argparse, csv, pickle
import numpy as np

WINDOWS = {  # state file stem: (first occurrence, first breeding)
    'Marylan': (1958, 1963), 'NCaroli': (1962, 1975), 'SCaroli': (1966, 1979), 'Vermont': (1968, 1976),
    'W_Virgi': (1971, 1975), 'Ohio': (1964, 1976), 'Tenness': (1972, 1980), 'Kentuck': (1977, 1981),
    'Alabama': (0, 1982), 'Arkansa': (1971, 1985), 'Indiana': (1958, 1981), 'Illinoi': (1971, 1982),
    'Michiga': (1972, 1981), 'Wiscons': (1972, 1986), 'Missour': (1974, 1983), 'Minneso': (1980, 1989),
    'Iowa': (1982, 1986), 'Ontario': (1972, 1978)}
BBS = '/home/breallis/houfin/scratch/houfin/data/bbs_2026_release'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    import rasterio
    from rasterio.warp import transform
    m = pickle.load(open(f'{a.input}/metadata.pkl', 'rb'))
    land = np.asarray(m['land_mask']).astype(bool); lr, lc = np.nonzero(land)
    routes = {}
    with open(f'{BBS}/Routes.csv', newline='', encoding='latin-1') as fh:
        for r in csv.DictReader(fh):
            try:
                routes[(int(r['CountryNum']), int(r['StateNum']), int(r['Route']))] = (float(r['Latitude']), float(r['Longitude']))
            except ValueError:
                pass
    names, occ, brd, cells = [], [], [], []
    with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
        for st, (o, b) in WINDOWS.items():
            keys = set()
            with open(f'{BBS}/States/{st}.csv', newline='', encoding='latin-1') as fh:
                for r in csv.DictReader(fh):
                    keys.add((int(r['CountryNum']), int(r['StateNum']), int(r['Route'])))
            cs = set()
            for k in keys:
                if k not in routes:
                    continue
                la, lo = routes[k]; xs, ys = transform('EPSG:4326', src.crs, [lo], [la]); r_, c_ = src.index(xs[0], ys[0])
                if not (0 <= r_ < land.shape[0] and 0 <= c_ < land.shape[1] and land[r_, c_]):
                    i = int(np.argmin((lr - r_) ** 2 + (lc - c_) ** 2)); r_, c_ = int(lr[i]), int(lc[i])
                cs.add((r_, c_))
            names.append(st); occ.append(o); brd.append(b); cells.append(sorted(cs))
    mx = max(len(c) for c in cells)
    R = np.zeros((len(cells), mx), int); C = np.zeros((len(cells), mx), int); M = np.zeros((len(cells), mx))
    for i, cs in enumerate(cells):
        for j, (r_, c_) in enumerate(cs):
            R[i, j], C[i, j], M[i, j] = r_, c_, 1.
    np.savez_compressed(a.out, names=np.asarray(names), occ=np.asarray(occ), breed=np.asarray(brd), rows=R, cols=C, mask=M)
    print({n: (o, b, len(c)) for n, o, b, c in zip(names, occ, brd, cells)})


if __name__ == '__main__':
    main()
