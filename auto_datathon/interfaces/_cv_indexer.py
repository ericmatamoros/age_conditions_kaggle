# -*- coding: utf-8 -*-
"""
Script with the class "GetIndexer" to obtain cross-validation indexes.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold

class GetCVIndexer:
    """Dedicated class to obtain cross-validation indexes."""

    def __init__(self, df) -> None:
        self._df = df

    # Leave one out index
    def get_index_loo(self):
        """Leave-One-Out cross-validation indexes. """
        loo = LeaveOneOut()
        
        cv_indices = []
        for train_index, test_index in loo.split(self._df):
            cv_indices.append( (train_index, test_index) )

        return cv_indices
        
    def get_index_kfolds(self, n_splits: int):
        """KFolds cross-validation index.

        :param n_splits: Number of folds.     
        """
        kf = KFold(n_splits=n_splits)
        cv_indices = []
        for train_index, test_index in kf.split(self._df):
            cv_indices.append( (train_index, test_index) )

        return cv_indices