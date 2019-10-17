from .models import BnnScalarRegressor, BnnBinaryClassifier

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedKFold
from rate.models import BnnScalarRegressor, BnnBinaryClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import logging
logger = logging.getLogger(__name__)

def cross_validate_bnn(layer_list, X, y, k,
                       init_args={}, fit_args={}, score_args={},
                       n_jobs=None):
    """Stratified K-fold cross-validation of the BNN models defined by the layers in `layer_list` on data X, y.
    The type of model is inferred from y.

    Args:
        layer list: list of lists, where each list contains a layer specification for a BNN.
        X: data matrix with shape (n_examples, n_variables)
        y: labels. The type of model (regression or classification) will be inferred from y
        k: integer specifying the number of folds to use in the cross-validation
        init_args: kwargs passed to the BNN __init__ method
        fit_args: kwargs passed to the fit method of the BNN
        score_args: kwargs passed to the score method of the BNN. Controls the scoring metric (accuracy or auc)
        n_jobs: number of workers to parallise the fits over. Not currently implemented.
    """

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y do not have the same number of observations")

    if len(layer_list)==1:
        raise ValueError("For cross-validation please provide more than one possible architecture")

    target_type = type_of_target(y)
    logger.info("Running cross-validation over {} BNNs with {} targets".format(len(layer_list), target_type))
    logger.debug("init_args: {}\n, fit_args: {}\n, score_args: {}\n".format(init_args, fit_args, score_args))
    if target_type=="binary":
        bnn_init = BnnBinaryClassifier
    elif target_type=="continuous":
        bnn_init = BnnScalarRegressor
    else:
        raise ValueError("Unsupported target type {}".format(target_type))
        
    val_score_grid = np.zeros((len(layer_list), k))
    
    for i, layers in enumerate(layer_list):
        logger.debug("Model #{} of {}".format(i, len(layer_list)))
        for j, (train_idxs, val_idxs) in enumerate(StratifiedKFold(n_splits=k, shuffle=True).split(X, y)):
            logger.debug("Fold #{} of {}".format(j, k))
            X_train, X_val = X[train_idxs], X[val_idxs]
            y_train, y_val = y[train_idxs], y[val_idxs]
            
            bnn = bnn_init(layers=layers, **init_args)
            bnn.fit(X_train, y_train, **fit_args)
            val_score_grid[i,j] = bnn.score(X_val, y_val, **score_args)
    
    # Refit the best model on the whole dataset
    best_model_idx = val_score_grid.mean(axis=1).argmax()
    logger.info("Model {} had the largest mean validation metric: {:.3f} pm {:.3f} (mean pm std over {} folds)".format(
        best_model_idx,
        val_score_grid.mean(axis=1)[best_model_idx],
        val_score_grid.std(axis=1)[best_model_idx],
        k))
    bnn_best = bnn_init(layer_list[best_model_idx], **init_args)
    bnn_best.fit(X, y, **fit_args)
    
    return bnn_best, val_score_grid

# def cross_validate_bnn_sklearn(
#     X, y, param_grid,
#     k, n_iter=None,
#     just_estimator=True, search_type="random",
#     init_args={}, fit_args={}, search_args={}):
#     """Run k-fold cross validation on a BNN with data X and y. The model type (classification
#     or regression) is inferred from y.
    
#     Args:
#         X: data matrix, array with shape (n_examples, n_variables)
#         y: labels, array with shape (n_examples, 1)
#         param_grid: dictionary of hyperparameter settings for the BNN. Must have format
#                     {'layers' : list of layers, 'opt_fn' : list of optimisers_fns}
#         k: the number of folds
#         n_iter: the number of models to try (random search only)
#         just_estimator: return the best estimator or the whole search object  (default True)
#         search_type: grid or random search (default random)
#         init_args: passed to BNN __init__ method
#         fit_args: passed to the search fit method
#         search_args: passed to search __init__ method

#     Returns:
#        Either the Bnn model itself or an sklearn search object, depending on the value of
#        just_estimator
#     """
#     logger.debug("Cross validating with X shape {}, y shape {} and search type {}".format(
#         X.shape, y.shape, search_type))
#     if X.shape[0] != y.shape[0]:
#         raise ValueError("Number of items in X and y must match")

#     if search_type=="grid" and n_iter is not None:
#         logger.warning("Grid search is exhaustive - n_iter will be ignored")
        
#     target_type = type_of_target(y)
#     logger.info("Running {} search for BNN with {} targets".format(search_type, target_type))
#     if target_type=="binary":
#         bnn_init = BnnBinaryClassifier
#     elif target_type=="continuous":
#         bnn_init==BnnScalarRegressor
#     else:
#         raise ValueError("Unknown target type {}".format(target_type))

#     keras_model = KerasClassifier(build_fn=lambda layers, opt_fn: bnn_init(layers=layers, optimiser_fn=opt_fn, **init_args))
    
#     if search_type=="grid":
#         grid = GridSearchCV(estimator=keras_model,
#                             param_grid=param_grid,
#                             cv=k,
#                             **search_args).fit(X, y, **fit_args)
#     elif search_type=="random":
#         grid = RandomizedSearchCV(estimator=keras_model,
#                                   param_distributions=param_grid,
#                                   cv=k,
#                                   n_iter=n_iter,
#                                   **search_args).fit(X, y, **fit_args)
#     else:
#         raise ValueError("{} is not a valid search type - choose grid or random".format(search_type))
#     if just_estimator:
#         return grid.best_estimator_.model
#     else:
#         return grid

