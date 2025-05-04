import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression, LogisticRegression

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model tarining completed.")
            return reg
        except Exception as e:
            logging.error(f'Error while training the model: {e}')
            raise e
        
class LogisticRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        try:
            clf = LogisticRegression(**kwargs)
            clf.fit(X_train, y_train)
            logging.info("Model training completed.")
            return clf
        except Exception as e:
            logging.error(f'Error while training the model: {e}')
            raise e