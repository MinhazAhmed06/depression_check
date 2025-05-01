import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPrepStrat

def get_data_for_test():
    try:
        df = pd.read_csv('data/student_depression_dataset.csv')
        df = df.sample(n=100)
        prepstrat = DataPrepStrat()
        data_cleaning = DataCleaning(df, prepstrat)
        df = data_cleaning.handle_data()
        df.drop(['Depression'], axis=1, inplace=True)
        result = df.to_json(orient='split')
        return result
    except Exception as e:
        logging.error(e)
        raise e
