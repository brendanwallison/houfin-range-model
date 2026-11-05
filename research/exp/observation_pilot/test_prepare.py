import gzip
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from observation_pilot.prepare import checklist,ebird,season


class PreparationTests(unittest.TestCase):
    def record(self,sid='S1',group='',complete='1'):
        return {'ALL SPECIES REPORTED':complete,'PROTOCOL CODE':'P21','OBSERVATION DATE':'2018-06-10',
                'LATITUDE':'45','LONGITUDE':'-75','DURATION MINUTES':'30','NUMBER OBSERVERS':'1',
                'EFFORT DISTANCE KM':'','TIME OBSERVATIONS STARTED':'08:30:00',
                'SAMPLING EVENT IDENTIFIER':sid,'GROUP IDENTIFIER':group,'OBSERVER ID':'observer',
                'LOCALITY ID':'location'}

    def test_winter_year_and_incomplete_denominator(self):
        self.assertEqual(season('2018-12-20'),('winter',2019))
        self.assertEqual(season('2019-01-03'),('winter',2019))
        self.assertIsNone(checklist(self.record(complete='0')))
        r=self.record();r['PROTOCOL CODE']='P22'
        self.assertIsNone(checklist(r))

    def test_full_scan_zero_filling_x_and_shared_lists(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'CA').mkdir()
            records=[self.record('S1','G1'),self.record('S2','G1'),self.record('S3'),self.record('S4')]
            sef=p/'CA/ebd_CA_relAug-2026_sampling.txt.gz'
            with gzip.open(sef,'wt') as f:pd.DataFrame(records).to_csv(f,sep='\t',index=False)
            obs=[dict(records[0],**{'COMMON NAME':'House Finch','CATEGORY':'species','APPROVED':'1','OBSERVATION COUNT':'2'}),
                 dict(records[1],**{'COMMON NAME':'House Finch','CATEGORY':'species','APPROVED':'1','OBSERVATION COUNT':'2'}),
                 dict(records[2],**{'COMMON NAME':'House Finch','CATEGORY':'species','APPROVED':'1','OBSERVATION COUNT':'X'})]
            with gzip.open(p/'CA/ebd_CA_relAug-2026.txt.gz','wt') as f:pd.DataFrame(obs).to_csv(f,sep='\t',index=False)
            out=p/'out';ebird(p,out,'CA')
            frame=pd.read_parquet(out/'breeding.parquet').set_index('event_key')
            self.assertEqual(len(frame),3)
            self.assertEqual(frame.loc['G:G1','count'],2)
            self.assertEqual(frame.loc['S:S3','detected'],1)
            self.assertTrue(pd.isna(frame.loc['S:S3','count']))
            self.assertEqual(frame.loc['S:S4','count'],0)
            self.assertEqual(frame.loc['S:S4','detected'],0)
            self.assertTrue(json.load(open(out/'READY.json'))['status']=='complete')


if __name__=='__main__':unittest.main()
