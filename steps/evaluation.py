import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

from zenml import step
from zenml.client import Client 
import mlflow

from sklearn.base import RegressorMixin, BaseEstimator
from src.eval import MSE, R2, RMSE


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, 'mse'],
    Annotated[float, 'r2'],
    Annotated[float, 'rmse']
]:
    try:
        prediction = model.predict(X_test)
        mse = MSE().calculate_score(y_test, prediction)
        mlflow.log_metric('mse', mse)
        r2 = R2().calculate_score(y_test, prediction)
        mlflow.log_metric('r2', r2)
        rmse = RMSE().calculate_score(y_test, prediction)
        mlflow.log_metric('rmse', rmse)
        return mse, r2, rmse
    except Exception as e:
        logging.error(f'Error while evaluating model: {e}')
        raise e