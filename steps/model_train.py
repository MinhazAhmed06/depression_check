import logging
import pandas as pd
from typing import Tuple

from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig
) -> RegressorMixin:
    try:
        model = None
        if config.selected_model == 'LinearRegression':
            model = LinearRegressionModel()
            trained_model = model.fit(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f'Model {config.selected_model} is not supported!')
    except Exception as e:
        logging.error(f'Error while training the model {e}')
        raise e
            