from keras.callbacks import EarlyStopping
from src.functions import target_predictors_separate

class TrainingData:
    def __init__(self, training, validation, test, target_col):
        training_targets, training_predictors = target_predictors_separate(training, training.columns, target_col)
        validation_targets, validation_predictors = target_predictors_separate(validation, validation.columns, target_col)
        test_targets, test_predictors = target_predictors_separate(test, test.columns, target_col)

        self.predictor_mean = training_predictors.mean()
        self.predictor_std = training_predictors.std().replace(0, 1)

        self.training_targets = training_targets
        self.training_predictors = (training_predictors - self.predictor_mean) / self.predictor_std

        self.validation_targets = validation_targets
        self.validation_predictors = (validation_predictors - self.predictor_mean) / self.predictor_std
        
        self.test_targets = test_targets
        self.test_predictors = (test_predictors - self.predictor_mean) / self.predictor_std


    @property
    def n_cols(self):
        return self.training_predictors.shape[1]

    def fit(self, model, epochs=100, early_stopping = False):
        es = None
        if early_stopping:
            es = [EarlyStopping(
                monitor = 'val_loss',
                patience = 3,
                mode = 'min',
                restore_best_weights=True
            )]
        return model.fit(
            self.training_predictors,
            self.training_targets,
            validation_data=(self.validation_predictors, self.validation_targets),
            epochs=epochs,
            callbacks=es
        )
    def evaluate(self, model):
        return model.evaluate(
                self.test_predictors,
                self.test_targets
        )