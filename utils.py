import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def dropColumnList(df,delCols):
    rdf= pd.DataFrame()
    for c in delCols:
        if c in df.keys().to_list():
            df.drop(c,axis=1,inplace=True)
    rdf=df

    return rdf


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