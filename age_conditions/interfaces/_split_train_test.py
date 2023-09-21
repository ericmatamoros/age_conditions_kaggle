# -*- coding: utf-8 -*-
"""
Script with the class "Splitter" to split the data between train-test
"""

import numpy as np
import pandas as pd

class TrainTestSplitter:
    """Class used to obtain the train-test split."""

    def __init__(self, df: pd.DataFrame, features: list, target: str, test_size: float) -> None:

        self._df = df
        self._features = features
        self._target = target
        self._test_size = test_size

    
    # Split by percentage with 
    def split_by_percentage(self):
        """Split by percentage of train-test. General approach. """

        df = self._df

        df['index'] = range(0, df.shape[0])
        train_rows = int(round(df.shape[0] * self._test_size, 0))

        train = df.sample(n = train_rows, replace=False)
        train_index = np.unique(train['index'])

        test = df[~df['index'].isin(train_index)]

        return train[self._features], train[self._target], test[self._features], test[self._target]

    def split_by_date_ts(self, fcst_split: pd.Timestamp, date_col: str):
        """Split timeseries by specific date."""

        df = self._df
        
        # Define train & test splits
        train = df[df[date_col] < fcst_split]
        test = df[df[date_col] >= fcst_split]
        
        return train[self._features], train[self._target], test[self._features], test[self._target]