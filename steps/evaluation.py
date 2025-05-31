import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

from zenml import step
from zenml.client import Client 
import mlflow

from sklearn.base import BaseEstimator
from src.eval import Precision, Recall, F1score


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, 'precision'],
    Annotated[float, 'recall'],
    Annotated[float, 'f1score']
]:
    try:
        prediction = model.predict(X_test)
        prec = Precision().calculate_score(y_test, prediction)
        mlflow.log_metric('precision', prec)
        rec = Recall().calculate_score(y_test, prediction)
        mlflow.log_metric('recall', rec)
        f1s = F1score().calculate_score(y_test, prediction)
        mlflow.log_metric('f1score', f1s)
        return prec, rec, f1s
    except Exception as e:
        logging.error(f'Error while evaluating model: {e}')
        raise e