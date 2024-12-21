import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import sqlalchemy
from sqlalchemy import text
from google.cloud import storage
#import cloudstorage as gcs
import json
import re

def dropColumnList(df,delCols):
    rdf= pd.DataFrame()
    for c in delCols:
        if c in df.keys().to_list():
            df.drop(c,axis=1,inplace=True)
    rdf=df

    return rdf


def fetchHeaders(engine,tableName,verbose=0):
    qry=sqlalchemy.text(
        f'SELECT a.attname as "Column" FROM pg_catalog.pg_attribute a WHERE a.attnum > 0 AND NOT a.attisdropped AND a.attrelid = (SELECT c.oid FROM pg_catalog.pg_class c LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE c.relname ~ \'{tableName}\' AND pg_catalog.pg_table_is_visible(c.oid));'
    )
    if(int(verbose)==1):
        print(qry)
    with engine.connect() as conn:
        try:
            resultset = conn.execute(qry)
            results_as_dict = resultset.mappings().all()
        except:
            results_as_dict=[]
        return results_as_dict
    conn.close()
    

def findHeaderCandidatesInData(engine,tableName,col,colst,regex='[a-z][\/\-\+]*',limit=2,print=1):
    #limit = 1 just returns the first result. But in most csv merges there are many left from each merged file. 
    #this code can be used to clean those up as well
    #qry=sqlalchemy.text(f'SELECT DISTINCT "{colst}", "{tableName}".* FROM "{tableName}" WHERE "{col}" ~ \'{keyword}\' limit {limit}')
    qry=sqlalchemy.text(f'SELECT * FROM "{tableName}" WHERE "{col}"::text ~ \'{regex}\' limit {limit}')
    if(int(print)==1):
        print(qry)
    with engine.connect() as conn:
        try:
            resultset = conn.execute(qry)
            results_as_dict = resultset.mappings().all()
        except:
            results_as_dict=[]
        return results_as_dict
    conn.close()


    #note the customized header references that require backout to front ui or config
def dfToCsvCloud(dataframe,uri,VERBOSE=True):
    client = storage.Client()
    if VERBOSE:
        print(f'\-\-\-Uploading to {uri}')
    match = re.match(r"gs://([^/]+)/(.+)/(.+)", uri)
    slash="/"
    try: 
        g2 = match.group(2)
    except: 
        g2= ''
        slash=''
    try:
        g3 = match.group(3)
    except:
        key=''
        g3=''
        slash=''
    bucket_path = f'{g2}{slash}{g3}_stats.csv'
    bucket_name = match.group(1)
    bucket = client.bucket(bucket_name)
    #bucket.blob(key + '/' + asset).upload_from_string(dataframe.to_csv(sep=separator,index=index,header=alias,encoding=encoding), 'text/csv')
    bucket.blob(bucket_path).upload_from_string(dataframe.to_csv(), 'text/csv')
    #bucket.blob(key + '/' + asset + '.csv').upload_from_string(dataframe.to_csv(sep=separator,index=ind,header=alias,encoding=encoding), 'text/csv')
