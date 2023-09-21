""" Definition of parameters for models """
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from ._errors import ModelNameInexistent

def obtain_exhaustive_grid(name: str) -> dict:
    """Get model-specific parameters that will be applied to a random HP exhaustive approach.

    :param name: Name of the model that wants to be used.
    :return: Dictionary with parameters for a specific model.
    """
    if name == 'xgboost':
        params = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
                }

    elif name == 'lgbm':
        params = {
            'num_leaves':[20,40,60,80,100],
            'min_child_samples':[5,10,15],
            'max_depth':[-1,5,10,20],
            'learning_rate':[0.05,0.1,0.2],
            'reg_alpha':[0,0.01,0.03]
            }

    else:
        raise ModelNameInexistent("Model name provided is invalid or inexistent parameters have been found")

    return params

def obtain_random_grid(name: str) -> dict:
    """Get model-specific parameters that will be applied to a random HP tunning approach.

    :param name: Name of the model that wants to be used.
    :return: Dictionary with parameters for a specific model.
    """
    if name == 'xgboost':
        params = {
                'min_child_weight': [1, 5, 10],
                'gamma': sp_randint(0.5, 10),
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': sp_randint(0.4, 1),
                'max_depth': sp_randint(1, 15)
                }
    elif name == 'lgbm':
        params ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

    else:
        raise ModelNameInexistent("Model name provided is invalid or inexistent parameters have been found")

    return params