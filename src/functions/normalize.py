def normalize(predictors):
    return (predictors - predictors.mean()) / predictors.std()
