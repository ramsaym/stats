def dropColumnList(df,delCols,keepCols):
    for c in delCols:
        if c in keepCols:
            df.drop(c)
    return df