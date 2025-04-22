import logging

import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrat, DataPrepStrat

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    try:
        prep_strat = DataPrepStrat()
        data_cleaning = DataCleaning(df, prep_strat)
        processed_data = data_cleaning.handle_data()
        
        divide_strat = DataDivideStrat()
        data_divided = DataCleaning(processed_data, divide_strat)
        X_train, X_test, y_train, y_test = data_divided.handle_data()

        logging.info('Data cleaning completed')
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(f'Error in cleaning data {e}')
        raise e

