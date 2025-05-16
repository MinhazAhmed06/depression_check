import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass
        
class precision(Evaluation):
    def calculate_score(self, y_true, y_pred):
        try:
            logging.info('Calculating precision.')
            precision = precision_score(y_true, y_pred, average='macro')
            logging.info(f'precision: {precision}')
            return precision
        except Exception as e:
            logging.error(f'Error while calculating precision {e}')
            raise e
        
class recall(Evaluation):
    def calculate_score(self, y_true, y_pred):
        try:
            logging.info('Calculating recall.')
            recall = recall_score(y_true, y_pred, average='macro')
            logging.info(f'recall: {recall}')
            return recall
        except Exception as e:
            logging.error(f'Error while calculating recall {e}')
            raise e
        
class f1score(Evaluation):
    def calculate_score(self, y_true, y_pred):
        try:
            logging.info('Calculating f1_score.')
            f1score = f1_score(y_true, y_pred, average='macro')
            logging.info(f'f1_score: {f1score}')
            return f1score
        except Exception as e:
            logging.error(f'Error while calculating f1_score {e}')
            raise e