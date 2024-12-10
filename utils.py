import pandas

def dropColumnList(df,delCols,keepCols):
    
    for c in delCols:
        if keepCols is not None:
            if c in keepCols:
                df.drop(c)
        else:
            if c in df.keys().to_list():
                df.drop(c)

    return df