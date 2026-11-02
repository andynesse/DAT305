from src.functions import normalize, target_predictors_separate

class TrainingData:
    def __init__(self, df, target_col):
        target, predictors = target_predictors_separate(df, df.columns, target_col)
        self.target = target
        self.predictors = predictors
        self.predictors_norm = normalize(predictors)

    @property
    def n_cols(self):
        return self.predictors_norm.shape[1]

    def fit(self, model, validation_split=0.3, epochs=100):
        return model.fit(
            self.predictors_norm,
            self.target,
            validation_split=validation_split,
            epochs=epochs
        )
