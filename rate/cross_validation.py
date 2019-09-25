from .models import BnnScalarRegressor, BnnBinaryClassifier

from sklearn.utils.multiclass import type_of_target
from rate.models import BnnScalarRegressor, BnnBinaryClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import logging
logger = logging.getLogger(__name__)

def cross_validate_bnn(
    X, y, param_grid,
    k, n_iter=None,
    just_estimator=True, search_type="random",
    init_args={}, fit_args={}, search_args={}):
    """Run k-fold cross validation on a BNN with data X and y. The model type (classification
    or regression) is inferred from y.
    
    Args:
        X: data matrix, array with shape (n_examples, n_variables)
        y: labels, array with shape (n_examples, 1)
        param_grid: dictionary of hyperparameter settings for the BNN. Must have format
                    {'layers' : list of layers, 'optimiser_fn' : list of optimisers_fns}
        k: the number of folds
        n_iter: the number of models to try (random search only)
        just_estimator: return the best estimator or the whole search object  (default True)
        search_type: grid or random search (default random)
        init_args: passed to BNN __init__ method
        fit_args: passed to the search fit method
        search_args: passed to search __init__ method

    Returns:
       Either the Bnn model itself or an sklearn search object, depending on the value of
       just_estimator
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of items in X and y must match")

    if search_type=="grid" and n_iter is not None:
        logger.warning("Grid search is exhaustive - n_iter will be ignored")
        
    target_type = type_of_target(y)
    if target_type=="binary":
        bnn_init = BnnBinaryClassifier
    elif target_type=="continuous":
        bnn_init==BnnScalarRegressor
    else:
        raise ValueError("Unknown target type {}".format(target_type))
        
    # def create_bnn(layers, optimiser_fn):
    #     bnn = bnn_init(layers=layers, optimiser_fn=optimiser_fn, **init_args)
    #     # bnn.fit(np.random.randn(n, p), np.random.choice(2, size=(n, 1), replace=True))
    #     return bnn

    keras_model = KerasClassifier(build_fn=lambda l, opt_fn: bnn_init(layers=l, optimiser_fn=opt_fn, **init_args))
    
    if search_type=="grid":
        grid = GridSearchCV(estimator=keras_model,
                            param_grid=param_grid,
                            cv=k,
                            n_iter=n_iter,
                            **search_args).fit(X, y, **fit_args)
    elif search_type=="random":
        grid = RandomizedSearchCV(estimator=keras_model,
                                  param_distributions=param_grid,
                                  cv=k,
                                  n_iter=n_iter,
                                  **search_args).fit(X, y, **fit_args)
    else:
        raise ValueError("{} is not a valid search type - choose grid or random".format(search_type))
    if just_estimator:
        return grid.best_estimator_.model
    else:
        return grid