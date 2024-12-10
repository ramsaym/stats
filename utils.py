import pandas

def dropColumnList(df,delCols,keepCols):
    
    for c in delCols:
        if keepCols is not None:
            if c in keepCols:
                rdf = df.drop(c,axis=1, inplace=True)
        else:
            if c in df.keys().to_list():
                rdf = df.drop(c,axis=1, inplace=True)

    return rdf