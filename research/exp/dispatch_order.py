"""Preserve batch priorities; interleave first scientific designs before replicas."""
import re

def dispatch_order(name):
 name=str(name).rsplit('/',1)[-1];priority=name.split('_',1)[0]
 if priority!='0609':return (priority,0,0,name)
 match=re.search(r'_s(41|42|43)\.sh$',name)
 seed=int(match.group(1))-41 if match else 99
 order=['step45_cell_iid_s010','step44_kprivate10','step45_regional_s010','step44_allrho99','step44_kprivate25',
        'step45_cell_iid_s005','step45_regional_s005','step45_cell_iid_s020','step45_regional_s020']
 rank=next((i for i,p in enumerate(order) if '_'+p+'_s' in name),99)
 return (priority,seed,rank,name)