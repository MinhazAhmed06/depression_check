import logging
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

from zenml import step

from sklearn.base import RegressorMixin
from src.eval import MSE, R2, RMSE

@step
def evaluate_model(
    model: RegressorMixin,
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
        r2 = R2().calculate_score(y_test, prediction)
        rmse = RMSE().calculate_score(y_test, prediction)
        return mse, r2, rmse
    except Exception as e:
        logging.error(f'Error while evaluating model: {e}')
        raise e