def target_predictors_separate(df, columns, target_col):
    target = df[target_col]
    predictors = df[columns[columns != target_col]]
    return target, predictors


