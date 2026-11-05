"""Audit annual reproductive conversion support and reusable survival inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    base=Path('/mnt/e/Datasets/houfin/nestwatch/mendeley_wjf794z7gc_v6/hofi')
    attempts=pd.read_csv(base/'hofi_attempts.csv');checks=pd.read_csv(base/'hofi_checks.csv')
    root=Path('/mnt/e/Datasets/houfin/grid54/nestwatch')
    with np.load(root/'nests_hofi_v6.npz') as z:arrays={k:z[k] for k in z}
    with np.load(root/'nests_hofi_v6_ids.npz') as z:ids=z['attempt_id']
    keep=(arrays['year']>=1902)&(arrays['year']<=2020)&(arrays['precision']<=1)&((arrays['has_exp']==1)|((arrays['outcome']==1)&(arrays['fledged']>=1)))&(arrays['t_idx']>=0)&(arrays['t_idx']<124)
    count=keep&(arrays['outcome']==1)&(arrays['fledged']>=1)
    exposure=keep&(arrays['has_exp']==1)
    # Repeated locations are reporting units, not known pairs or known females.
    loc=attempts.groupby(['Location ID','Year']).size()
    usable=pd.DataFrame({'year':arrays['year'][keep],'region':arrays['region'][keep],
                         'substrate':arrays['substrate'][keep],'source':arrays['source'][keep],
                         'exposure':exposure[keep],'count':count[keep]})
    coverage=usable.groupby(['year','region']).agg(n=('year','size'),exposure=('exposure','sum'),count=('count','sum')).reset_index()
    coverage.to_parquet(a.out/'nest_training_coverage.parquet',index=False)
    absent=[name for name in ['Female ID','Pair ID','Band ID','Season Complete'] if name not in attempts.columns]
    result={'raw_attempt_rows':len(attempts),'raw_unique_attempts':int(attempts['Attempt ID'].nunique()),
      'on_land_attempts':len(ids),'likelihood_attempts':int(keep.sum()),'exposure_rows':int(exposure.sum()),'count_rows':int(count.sum()),
      'share_training_since_2000':float((arrays['year'][keep]>=2000).mean()),
      'locations':int(attempts['Location ID'].nunique()),'reported_location_years':len(loc),
      'location_years_with_multiple_reported_attempts':int((loc>1).sum()),
      'missing_individual_and_season_denominator_fields':absent,
      'annual_attempts_identified':False,
      'interpretation':'Repeated reported attempts at a location do not identify attempts per female, breeding participation, or missed attempts. No complete-season female denominator was found in these schemas.',
      'source_fields':list(attempts.columns),'visit_fields':list(checks.columns)}
    write_json(a.out/'nest_conversion_audit.json',result)
    pilot=Path('/mnt/c/Dev/houfin-range-model/.research_runtime/bbl_adult_pilot_20261031')
    with np.load(pilot/'adult_marray.npz') as z:
        counts=z['counts'];denominator=counts.sum((1,2));deaths=counts[:,:,:-1].sum((1,2))
    write_json(a.out/'bbl_integration_inventory.json',{'regions':['west','plains','central_east','east'],
      'bands_by_region':denominator.tolist(),'dead_recoveries_by_region':deaths.tolist(),
      'all_bands':int(denominator.sum()),'all_dead_recoveries':int(deaths.sum()),
      'coarse_differentiable_block':'research/exp/observation_pilot/bbl_block.py',
      'production_spatial_adapter_complete':False,
      'remaining_contracts':['full adult movement-plus-destination-survival operator including native departure',
        'age/census and partial-exposure agreement','regional constant-survival approximation versus propagated cohort subsets',
        'reporting/delayed-recovery sensitivity','no duplicate event likelihood for movement and survival']})
    print('NEST',result['likelihood_attempts'],'BBL',int(denominator.sum()),int(deaths.sum()))


if __name__=='__main__':main()
