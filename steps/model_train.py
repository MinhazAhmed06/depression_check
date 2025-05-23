import logging
import pandas as pd
from typing import Tuple, Union

import mlflow
from zenml import step
from zenml.client import Client

from src.model_dev import LinearRegressionModel, LogisticRegressionModel
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model._logistic import LogisticRegression
from .config import ModelNameConfig


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig
) -> BaseEstimator:
    try:
        model = None
        if config.selected_model == 'LinearRegression':
            mlflow.sklearn.autolog()
            config = config.params
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train, **config)
            return trained_model
        elif config.selected_model == 'LogisticRegression':
            mlflow.sklearn.autolog()
            config = config.params
            model = LogisticRegressionModel()
            trained_model = model.train(X_train, y_train, **config)
            return trained_model
        else:
            raise ValueError(f'Model {config.selected_model} is not supported!')
    except Exception as e:
        logging.error(f'Error while training the model {e}')
        raise e
            