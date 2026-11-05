"""Separate focal House Finch observations; never used for DESK training.

Complete source scans precede zero-filled products. Event/group identifiers are
retained so shared eBird lists are one sampling unit, including presence-only X.
"""
import argparse
from collections import Counter
from datetime import date
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def identity(path):
    stat=path.stat()
    return {'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def season(value):
    d=date.fromisoformat(value)
    if 145<=d.timetuple().tm_yday<=196:return 'breeding',d.year
    if (d.month==12 and d.day>=14) or (d.month==1 and d.day<=5):
        return 'winter',d.year+(d.month==12)
    return None,None


def checklist(record):
    if record['ALL SPECIES REPORTED']!='1' or record['PROTOCOL CODE'] not in ('P21','P22'):return None
    try:
        s,y=season(record['OBSERVATION DATE'])
        lat,lon=float(record['LATITUDE']),float(record['LONGITUDE'])
        duration=float(record['DURATION MINUTES']); observers=int(record['NUMBER OBSERVERS'])
        distance=float(record['EFFORT DISTANCE KM'] or '0')
    except (ValueError,TypeError):return None
    if s is None or not 2010<=y<=2025 or not 24<=lat<=53 or not -125<=lon<=-66:return None
    if not 0<duration<=180 or not 0<=distance<=5 or not 1<=observers<=10:return None
    # A traveling checklist with unknown distance is not equivalent to stationary.
    if record['PROTOCOL CODE']=='P22' and record['EFFORT DISTANCE KM']=='':return None
    clock=record['TIME OBSERVATIONS STARTED']
    try:
        h,m,*_=clock.split(':'); hour=int(h)+int(m)/60
        if not 0<=hour<24:return None
    except (ValueError,TypeError):return None
    sid=record['SAMPLING EVENT IDENTIFIER'];group=record['GROUP IDENTIFIER']
    key=('G:'+group) if group else ('S:'+sid)
    return (key,sid,group,record['OBSERVER ID'],record['LOCALITY ID'],lat,lon,
            record['OBSERVATION DATE'],s,int(y),hour,record['PROTOCOL CODE'],duration,distance,observers)


def ebird(raw,out,country):
    out.mkdir(parents=True,exist_ok=True)
    done=out/'READY.json'
    sampling=raw/country/f'ebd_{country}_relAug-2026_sampling.txt.gz'
    source=raw/country/f'ebd_{country}_relAug-2026.txt.gz'
    sources=[identity(sampling),identity(source)]
    if done.exists():
        if json.loads(done.read_text())['sources']!=sources:raise ValueError('completed source identity changed')
        return
    database=out/'events.sqlite'
    if database.exists():raise FileExistsError('partial extract retained; use a separate output for a new attempt')
    con=sqlite3.connect(database)
    con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA synchronous=NORMAL')
    con.execute('''CREATE TABLE events(event_key TEXT PRIMARY KEY,checklist_id TEXT,group_id TEXT,
      observer_id TEXT,locality_id TEXT,lat REAL,lon REAL,date TEXT,season TEXT,year INTEGER,
      hour REAL,protocol TEXT,duration_min REAL,distance_km REAL,n_observers INTEGER,
      detected INTEGER DEFAULT 0,count REAL DEFAULT 0,x_only INTEGER DEFAULT 0)''')
    columns=['ALL SPECIES REPORTED','PROTOCOL CODE','OBSERVATION DATE','LATITUDE','LONGITUDE',
             'DURATION MINUTES','NUMBER OBSERVERS','EFFORT DISTANCE KM','TIME OBSERVATIONS STARTED',
             'SAMPLING EVENT IDENTIFIER','GROUP IDENTIFIER','OBSERVER ID','LOCALITY ID']
    n=0;eligible=0;started=time.time()
    for chunk in pd.read_csv(sampling,sep='\t',usecols=columns,dtype=str,keep_default_na=False,chunksize=100000):
        n+=len(chunk)
        # Vectorized coarse rejection avoids date parsing for most of the SEF.
        chunk=chunk[(chunk['ALL SPECIES REPORTED']=='1')&chunk['PROTOCOL CODE'].isin(['P21','P22'])]
        records=[v for r in chunk.to_dict('records') if (v:=checklist(r)) is not None]
        eligible+=len(records)
        con.executemany('INSERT OR IGNORE INTO events VALUES ('+','.join(['?']*15)+',0,0,0)',records)
        con.commit()
        write_json(out/'progress.json',{'phase':'sampling_events','rows_read':n,'eligible_before_group_dedup':eligible,
                                        'epoch':time.time(),'elapsed_seconds':time.time()-started})
    con.execute('CREATE INDEX event_season ON events(season,year)');con.commit()
    print(country,'sampling scan complete',n,'rows; eligible',eligible,flush=True)
    statements=[];matches=0;lines=0;conflicts=0
    # Full EBD is very large. Parse only focal lines; exact fields reject matches in comments.
    with gzip.open(source,'rb') as stream:
        header=stream.readline().decode('utf-8-sig').rstrip('\r\n').split('\t')
        idx={k:i for i,k in enumerate(header)}
        for line in stream:
            lines+=1
            if b'\tHouse Finch\t' in line:
                values=line.decode('utf-8').rstrip('\r\n').split('\t')
                if values[idx['COMMON NAME']]!='House Finch' or values[idx['CATEGORY']]!='species' or values[idx['APPROVED']]!='1':continue
                group=values[idx['GROUP IDENTIFIER']];sid=values[idx['SAMPLING EVENT IDENTIFIER']]
                key=('G:'+group) if group else ('S:'+sid)
                text=values[idx['OBSERVATION COUNT']]
                number=None if text=='X' else float(text)
                if number is not None and (number<0 or not math.isfinite(number)):raise ValueError('invalid count')
                statements.append((number,number,int(number is None),key));matches+=1
            if len(statements)>=5000:
                # Shared copies are not additive. Conflicting shared counts are flagged in metadata.
                for number,_,x,key in statements:
                    old=con.execute('SELECT detected,count FROM events WHERE event_key=?',(key,)).fetchone()
                    if old and old[0] and number is not None and old[1] not in (None,number):conflicts+=1
                con.executemany('''UPDATE events SET detected=1,
                    count=CASE WHEN ? IS NULL THEN CASE WHEN detected=0 THEN NULL ELSE count END
                               WHEN count IS NULL OR ? > count THEN ? ELSE count END,
                    x_only=CASE WHEN ? IS NULL AND (detected=0 OR count IS NULL) THEN 1 ELSE 0 END WHERE event_key=?''',
                    [(a,a,a,a,k) for a,_,x,k in statements])
                con.commit();statements=[]
            if lines%2000000==0:
                write_json(out/'progress.json',{'phase':'focal_observations','rows_read':lines,'focal_approved_rows':matches,
                    'compressed_bytes_read':stream.fileobj.tell(),'compressed_size':source.stat().st_size,
                    'epoch':time.time(),'elapsed_seconds':time.time()-started})
        if statements:
            con.executemany('''UPDATE events SET detected=1,
              count=CASE WHEN ? IS NULL THEN CASE WHEN detected=0 THEN NULL ELSE count END
                         WHEN count IS NULL OR ? > count THEN ? ELSE count END,
              x_only=CASE WHEN ? IS NULL AND (detected=0 OR count IS NULL) THEN 1 ELSE 0 END WHERE event_key=?''',
              [(a,a,a,a,k) for a,_,x,k in statements]);con.commit()
    if [identity(sampling),identity(source)]!=sources:raise ValueError('source changed during extraction')
    counts=pd.read_sql_query('SELECT season,year,COUNT(*) n,SUM(detected) detections,SUM(x_only) x_only FROM events GROUP BY season,year',con)
    for s in ('breeding','winter'):
        # Chunked export keeps peak memory independent of the country extract size.
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer=None
        for frame in pd.read_sql_query('SELECT * FROM events WHERE season=?',con,params=[s],chunksize=100000):
            frame['count']=frame['count'].astype(float)
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(out/f'{s}.parquet',table.schema)
            writer.write_table(table)
        if writer is not None:writer.close()
    con.close()
    write_json(done,{'status':'complete','country':country,'sources':sources,'sampling_rows':n,'observation_rows':lines,
      'approved_focal_rows':matches,'shared_numeric_count_conflicts':conflicts,'counts':counts.to_dict('records'),
      'notes':['X is detection with unknown count, never an inferred zero.',
               'Shared-group numeric conflicts use maximum and are reported; sensitivity required.',
               'Winter year is January end year; seasonal projection census remains a separate contract.']})
    print(country,'READY',counts.to_dict('records'),flush=True)


def cbc(raw,out,year_min=1940,year_max=2025):
    out.mkdir(parents=True,exist_ok=True)
    files={k:next(raw.glob(f'*CBC_{k}_Report.csv')) for k in ('Circle_Species','Effort_Many_Types','Effort_Summary','Weather')}
    if (out/'READY.json').exists():return
    frames=[];focal=[];sources={k:identity(p) for k,p in files.items()}
    cols=['abbrev','latitude','longitude','country_code','count_yr','cnt_dt','com_name','how_many']
    for chunk in pd.read_csv(files['Circle_Species'],encoding='utf-8-sig',usecols=cols,chunksize=200000):
        chunk=chunk[chunk.country_code.isin(['US','CA'])&chunk.latitude.between(24,53)&chunk.longitude.between(-125,-66)&((chunk.count_yr+1900)>=year_min)&((chunk.count_yr+1900)<=year_max)]
        frames.append(chunk.drop(columns=['com_name','how_many']).drop_duplicates(['abbrev','count_yr']))
        focal.append(chunk[chunk.com_name=='House Finch'][['abbrev','count_yr','how_many']])
    events=pd.concat(frames,ignore_index=True).drop_duplicates(['abbrev','count_yr'])
    detections=pd.concat(focal,ignore_index=True)
    detections['count']=pd.to_numeric(detections.how_many,errors='coerce')
    text=detections.how_many.astype(str).str.lower()
    week=text.isin(['cw','count week','count-week']);detections.loc[week,'count']=0
    detections['unknown_count']=detections['count'].isna()
    totals=detections.groupby(['abbrev','count_yr']).agg(count=('count',lambda x:x.sum(min_count=1)),unknown_count=('unknown_count','max'))
    events=events.merge(totals,on=['abbrev','count_yr'],how='left')
    # Only absence of any focal row is zero-filled. Unknown reported counts remain unknown.
    absent=events.unknown_count.isna();events.loc[absent,'count']=0;events.loc[absent,'unknown_count']=False
    events['year_end']=1900+events.count_yr
    events['date']=pd.to_datetime(events.cnt_dt,format='%m/%d/%Y %H:%M:%S',errors='coerce')
    events['date_year_mismatch']=events.date.dt.year+(events.date.dt.month>=7)-events.year_end
    effort=pd.read_csv(files['Effort_Many_Types'],encoding='utf-8-sig')
    hours=effort.groupby(['abbrev','count_yr']).hours.sum(min_count=1).rename('party_hours')
    events=events.merge(hours,on=['abbrev','count_yr'],how='left')
    events.loc[events.party_hours<=0,'party_hours']=np.nan
    summary=pd.read_csv(files['Effort_Summary'],encoding='utf-8-sig')
    summary=summary.groupby(['abbrev','count_yr'])[['field_counters','max_parties','feeder_hrs']].max()
    events=events.merge(summary,on=['abbrev','count_yr'],how='left')
    weather=pd.read_csv(files['Weather'],encoding='utf-8-sig').groupby(['abbrev','count_yr'])[['min_temp','max_temp','min_wind','max_wind']].first()
    events=events.merge(weather,on=['abbrev','count_yr'],how='left')
    events.to_parquet(out/'circle_years.parquet',index=False)
    coverage=events.assign(decade=(events.year_end//10)*10).groupby('decade').agg(n=('abbrev','size'),measured_effort=('party_hours',lambda x:x.notna().mean()),positive=('count',lambda x:(x>0).sum()))
    write_json(out/'READY.json',{'status':'complete','sources':sources,'requested_year_min':year_min,'requested_year_max':year_max,'circle_years':len(events),'circles':events.abbrev.nunique(),
       'year_min':int(events.year_end.min()),'year_max':int(events.year_end.max()),'coverage':coverage.reset_index().to_dict('records'),
       'date_year_mismatches':int(events.date_year_mismatch.fillna(1).ne(0).sum()),
       'unknown_count_rows':int(events.unknown_count.sum()),'count_week_rows':int(week.sum())})
    print('CBC READY',len(events),'circle-years',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',choices=['ebird','cbc']);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--country',choices=['CA','US'])
    a=p.parse_args()
    if a.source=='ebird':ebird(a.raw,a.out,a.country)
    else:cbc(a.raw,a.out)


if __name__=='__main__':main()
