import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def dropColumnList(df,delCols):
    rdf= pd.DataFrame()
    for c in delCols:
        if c in df.keys().to_list():
            df.drop(c,axis=1,inplace=True)
    rdf=df

    return rdf