"""One-line summaries of structural_fit_readout.py outputs: potential-lambda medians by region/year, disease mortality, held-out.
  python readout_brief.py NAME=DIR/readout.json [NAME=...]"""
import json, sys
for arg in sys.argv[1:]:
    name, path = arg.split('=', 1)
    r = json.load(open(path))
    parts = []
    for reg in ('east', 'plains', 'west'):
        ys = r['regions'][reg]['years']
        pot = {y: round(v['potential_lambda_q10_50_90'][1], 2) for y, v in ys.items() if y in ('1966', '1990', '2010', '2020')}
        dm = {y: round(v['disease_mortality_q10_50_90'][1], 3) for y, v in ys.items() if y in ('2010', '2020')}
        parts.append(f"{reg[0].upper()} pot {pot} dis {dm}")
    print(f"{name:18s} " + ' | '.join(parts))
