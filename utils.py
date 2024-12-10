def dropColumnList(df,cols):
    for c in cols:
        if c in df.columns:
            df = df.drop(c)
    return df