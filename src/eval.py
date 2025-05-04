import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE.')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'MSE: {mse}')
            return mse
        except Exception as e:
            logging.error(f'Error while calculating MSE {e}')
            raise e
        
class R2(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating R2 score.')
            r2 = r2_score(y_true, y_pred)
            logging.info(f'R2 score: {r2}')
            return r2
        except Exception as e:
            logging.error(f'Error while calculating R2 score {e}')
            raise e
        
class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating RMSE.')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f'RMSE: {rmse}')
            return rmse
        except Exception as e:
            logging.error(f'Error while calculating RMSE {e}')
            raise e