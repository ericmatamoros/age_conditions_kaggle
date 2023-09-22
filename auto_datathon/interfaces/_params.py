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
            'n_estimators': [100, 200, 300],        # Number of boosting rounds (trees)
            'learning_rate': [0.01, 0.1, 0.2],     # Step size shrinkage to prevent overfitting
            'max_depth': [3, 4, 5],                # Maximum depth of individual trees
            'min_child_weight': [1, 2, 3],         # Minimum sum of instance weight needed in a child
            'gamma': [0, 0.1, 0.2],                # Minimum loss reduction required to make a further partition
            'subsample': [0.8, 0.9, 1.0],          # Fraction of samples used for fitting the trees
            'colsample_bytree': [0.8, 0.9, 1.0],   # Fraction of features used for building each tree
            'lambda': [0, 1, 2],                   # L2 regularization term on weights
            'alpha': [0, 1, 2],                    # L1 regularization term on weights
            'random_state': [42]                  # Seed for random number generation
        }

    elif name == 'lgbm':
        params = {
            'n_estimators': [100, 200, 300],          # Number of boosting rounds (trees)
            'learning_rate': [0.01, 0.1, 0.2],       # Step size shrinkage to prevent overfitting
            'max_depth': [3, 4, 5, 6],               # Maximum depth of individual trees
            'num_leaves': [15, 31, 63],              # Maximum number of leaves in one tree
            'min_child_samples': [1, 5, 10],         # Minimum number of data needed in a leaf
            'subsample': [0.8, 0.9, 1.0],            # Fraction of samples used for fitting the trees
            'colsample_bytree': [0.8, 0.9, 1.0],     # Fraction of features used for building each tree
            'reg_alpha': [0, 1, 2],                  # L1 regularization term on weights
            'reg_lambda': [0, 1, 2],                 # L2 regularization term on weights
            'random_state': [42]                    # Seed for random number generation
        }
        
    elif name == 'random_forest':
        params = {
            'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
            'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
            'max_depth': [None, 10, 20, 30],   # Maximum depth of the trees (None for no limit)
            'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],    # Minimum samples required to be at a leaf node
            'max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider for the best split
            'random_state': [42]              # Seed for random number generation
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