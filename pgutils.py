import pandas as pd
import numpy as np
import requests
import datetime
import csv
import os
import os.path
import sys
import glob
import sqlalchemy
from google.cloud.sql.connector import Connector
import sys
import pg8000
from sqlalchemy import text
import psycopg2
import gcsfs
from google.cloud import storage
from utils import *
from tqdm import tqdm
import math
from io import StringIO
import datetime as dt
from pgutils import *


def fetchHeaders(engine,tableName,verbose=1):
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
    
 ########################TO COMBINE############################################
def dfTosql(df, table_name, engine, chunksize=25000):
    total_rows = len(df)
    df['uid'] = df.index
    df['timestamp'] = pd.Series([dt.datetime.now()] * len(df))
    print("--DEBUG--Row Count: " + str(total_rows))
    with tqdm(total=total_rows) as pbar:
        for i in range(0, total_rows, chunksize):
            #trial for respect schema
            records = df.iloc[i:i+chunksize].to_sql(table_name,engine, if_exists='append', index=False)
            #records = pd.to_sql(df.iloc[i:i+chunksize], table_name, if_exists='append')

            pbar.set_description("Record count: " + str(records))
            pbar.update(chunksize)

def dfTosqlSync(df, table_name, engine, offset=0,chunksize=5000):
    total_rows = len(df) - offset
    print("--DEBUG--Row Count: " + str(total_rows))
    with tqdm(total=total_rows) as pbar:
        for i in range(0, total_rows, chunksize):
            iy=i+offset
            records = df.iloc[iy:iy+chunksize].to_sql(table_name, engine,schema='agiot', if_exists='append', index=False)
            pbar.set_description("Record count: " + str(records))
            pbar.update(chunksize)