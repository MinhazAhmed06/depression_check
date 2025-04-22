import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrat(ABC):
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPrepStrat(DataStrat):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data[data['Profession'] == 'Student']
            data = data.drop(['id','City','Profession','Work Pressure','Job Satisfaction','Degree'], axis=1)
            data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
            data = data[data['Sleep Duration'] != 'Others']
            data['Sleep Duration'] = data['Sleep Duration'].apply(
                        lambda x: (3 if 'More than 8 hours' in x 
                                else 2 if '7-8 hours' in x 
                                else 1 if '5-6 hours' in x 
                                else 0))
            data = data[data['Dietary Habits'] != 'Others']
            data['Dietary Habits'] = data['Dietary Habits'].apply(
                        lambda x: (2 if 'Healthy' in x 
                                else 1 if 'Moderate' in x 
                                else 0))
            data['Have you ever had suicidal thoughts ?'] = data['Have you ever had suicidal thoughts ?'].apply(lambda x:1 if x == 'Yes' else 0)
            data['Family History of Mental Illness'] = data['Family History of Mental Illness'].apply(lambda x:1 if x == 'Yes' else 0)
            return data

        except Exception as e:
            logging.error(f'Error while data preproecessing : {e}')
            raise e

class DataDivideStrat(DataStrat):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            x = data.drop(['Depression'], axis=1)
            y = data['Depression']
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error while dividing data: {e}')
            raise e

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrat):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_data(self.data)