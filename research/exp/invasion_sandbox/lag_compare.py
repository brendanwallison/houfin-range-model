"""Model 'mating common' year (state_lag.py) vs the documented FIRST BREEDING year per state (literature compiled by a research
agent 2026-11-06 from quoted sources; see SANDBOX_INVASION_SHAPE_2026-11.md ESTABLISHMENT-LAG LIT). Prints per-state offsets
(model - literature) for theta .25 / .5 / .75 and each fit, and the median / mean absolute offset."""
import json, sys
import numpy as np
LIT = {'Marylan': 1963, 'NCaroli': 1975, 'SCaroli': 1979, 'Vermont': 1976, 'W_Virgi': 1975, 'Ohio': 1976, 'Tenness': 1980,
       'Kentuck': 1981, 'Alabama': 1982, 'Arkansa': 1985, 'Indiana': 1981, 'Illinoi': 1982, 'Michiga': 1981, 'Wiscons': 1986,
       'Missour': 1983, 'Minneso': 1989, 'Iowa': 1986, 'Ontario': 1978}
O = '/mnt/e/Datasets/houfin/grid54/sandbox/ROUTEMARG'
fits = sys.argv[1:] or ['n21', 'pinned', 'n35']
D = {f: json.load(open(f'{O}/statelag_{f}.json'))['states'] for f in fits}
print('state     lit  | ' + ' | '.join(f'{f}: m.25 m.5 m.75' for f in fits) + ' | data1stBBS')
for s, ly in LIT.items():
    row = [f"{D[f][s]['mate_year']['0.25']} {D[f][s]['mate_year']['0.5']} {D[f][s]['mate_year']['0.75']}" for f in fits]
    print(f'{s:9s} {ly} | ' + ' | '.join(row) + f" | {D[fits[0]][s]['data_first_bbs']}")
for th in ('0.25', '0.5', '0.75'):
    for f in fits:
        d = np.array([D[f][s]['mate_year'][th] - ly for s, ly in LIT.items() if D[f][s]['mate_year'][th]])
        print(f'theta {th} {f:7s}: median offset {np.median(d):+.0f} yr, mean |offset| {np.mean(np.abs(d)):.1f}, within +-3 yr {np.mean(np.abs(d) <= 3):.2f}, n {d.size}')
