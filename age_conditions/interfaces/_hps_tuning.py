""" Class for hyperparameter tunning."""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ._params import obtain_exhaustive_grid, obtain_random_grid

class HyperTunner:
    """Class to perform the hyperparameter tuning."""

    def __init__(self,
                 estimator: object, 
                 hypers_grid: dict,
                 X_train: pd.DataFrame, 
                 y_train: pd.DataFrame, 
                 cv_indices: Tuple,
                 scorer: object,
                 verbose: int) -> None:
        self._estimator = estimator
        self._hypers_grid = hypers_grid
        self._X_train = X_train
        self._y_train = y_train
        self._cv_indices = cv_indices
        self._scorer = scorer
        self._verbose = verbose

    def exhaustive_tunner(self, n_jobs: int = 2):
        """Exhaustive search over specified parameter values for an estimator.

        :param n_jobs: Number of cores for parallelization.
        """
        model_grid = GridSearchCV(
            estimator= self._estimator,
            param_grid= self._hypers_grid,
            cv= self._cv_indices,
            verbose=self._verbose,
            n_jobs=n_jobs,
            scoring= self._scorer,
        )

        model_grid.fit(self._X_train.values, self._y_train.values)

        return model_grid.best_params_

    def random_tunner(self, n_jobs: int = 2):
        """Random search over distributed parameters for an estimator.

        :param n_jobs: Number of cores for parallelization.
        """

        # Random Grid Search
        model_grid = RandomizedSearchCV(
            estimator= self._estimator,
            param_grid= self._hypers_grid,
            cv= self._cv_indices,
            verbose=self._verbose,
            n_jobs=n_jobs,
            scoring= self._scorer,
        )

        model_grid.fit(self._X_train.values, self._y_train.values)

        return model_grid.best_params_

    def bayesian_tunner(self, n_jobs: int = 2):
        """Bayesian optimizer for a particular estimator.

        :param n_jobs: Number of cores for parallelization.
        """

        # Bayesian tunner
        model_grid = RandomizedSearchCV(
            estimator= self._estimator,
            search_spaces= self._hypers_grid,
            cv= self._cv_indices,
            verbose=self._verbose,
            n_jobs=n_jobs,
            scoring= self._scorer,
        )

        model_grid.fit(self._X_train.values, self._y_train.values)

        return model_grid.best_params_