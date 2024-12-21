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
    level=0
   
    try: 
        #3 level
        match = re.match(r"gs://([a-zA-Z]+)/([a-zA-Z]+)/([a-zA-Z]+)", uri)
        level=3
    except:
        #2 level
        try: 
            match = re.match(r"gs:\/\/([a-zA-Z]+)\/([a-zA-Z]+)",uri)
            level=2
        except:
            #single level bucket
            match = re.match(r"gs:\/\/([a-zA-Z]+)",uri)
            level=1

    slash="/"
    if (level==1):
        m1 = match.group(1)
        bucket_name = m1
        bucket_path = f'{bucket_name}_stats.csv'
        if (level==2):
            m2 = match.group(2)
            bucket_path = f'{m2}{slash}_stats.csv'
        else:
            m3 = match.group(3)
            bucket_path = f'{m2}{slash}{m3}{slash}_stats.csv'
    else:
        bucket_name=None    
    if (bucket_name is not None):
        if VERBOSE:
            print(f'---Uploading to {uri}')
        bucket = client.bucket(bucket_name)
        bucket.blob(bucket_path).upload_from_string(dataframe.to_csv(), 'text/csv')
  
