from .base import BaseDatasource
import pandas as pd


class TabularDatasource(BaseDatasource):
    """
    Generic purpose datasource object for tabular data processing
    """

    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe
    
    def clean_data(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx):
        return self._df[idx]