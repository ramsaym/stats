def dropColumnList(df,delCols,keepCols):
    for c in delCols:
        if c in keepCols:
            df.drop(c,axis=1, inplace=True)
    return df