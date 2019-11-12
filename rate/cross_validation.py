from .models import BnnScalarRegressor, BnnBinaryClassifier

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import StratifiedKFold
from rate.models import BnnScalarRegressor, BnnBinaryClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd

import random

import logging
logger = logging.getLogger(__name__)

def cross_validate_bnn(
    layer_list, optimizer_fn_list,
    search_type,
    X, y, k,
    n_iter=None,
    init_args={}, fit_args={}, score_args={},
    n_jobs=1):
    """Stratified K-fold cross-validation of the BNN models defined by the layers in `layer_list` on data X, y.
    The type of model is inferred from y.

    Args:
        layer list: list of lists, where each list contains a layer specification for a BNN.
        optimizer_fn_list: list of callables that return an optimizer
        search_type: one of `grid` or `random`, defining whether to do a grid or random search
        X: data matrix with shape (n_examples, n_variables)
        y: labels. The type of model (regression or classification) will be inferred from y
        k: integer specifying the number of folds to use in the cross-validation
        n_iter: number of search iterations (random search only)
        init_args: kwargs passed to the BNN __init__ method. Don't put any layers or optimiser_fn arguments here.
        fit_args: kwargs passed to the BNN fit method.
        score_args: kwargs passed to the BNN score method. Controls the scoring metric (accuracy or auc).
        n_jobs: number of workers to parallise the fits over (default 1, single worker). ***Not currently implemented***

    Returns:
        The best-performing BNN (in terms of the mean scoring metric over the k folds) and an array of scores for each
        model at each fold.
    """

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y do not have the same number of observations")

    if len(layer_list)==1:
        raise ValueError("For cross-validation please provide more than one possible architecture")

    target_type = type_of_target(y)
    logger.info("Running {} search cross-validation with {} targets".format(search_type, target_type))
    logger.debug("init_args: {}\n, fit_args: {}\n, score_args: {}\n".format(init_args, fit_args, score_args))
    if target_type=="binary":
        bnn_init = BnnBinaryClassifier
    elif target_type=="continuous":
        bnn_init = BnnScalarRegressor
    else:
        raise ValueError("Unsupported target type {}".format(target_type))

    if not np.all([callable(fn) for fn in optimizer_fn_list]):
        raise ValueError("all optimizers must be callable")
    
    param_list = [(layers, opt_fn) for layers in layer_list for opt_fn in optimizer_fn_list]

    if search_type=="grid":
        logger.info("Performing grid hyperparameter search with {} models".format(len(param_list)))
    elif search_type=="random":
        if n_iter is None:
            raise ValueError("For random search please select a number of parameter settings to sample")
        elif n_iter==len(param_list):
            logger.warning("n_iter is equal to the maximum number of combinations - performing grid search.")
        elif n_iter > len(param_list):
            raise ValueError("Maximum possible n_iter is {} for these layers/optimizers".format(len(param_list)))
        elif n_iter < len(param_list):
            logger.info("Performing random hyperparameter search with {} sampled models".format(n_iter))
            param_list = random.sample(param_list, n_iter)
    else:
        raise ValueError("Unrecognised search type {}, choose `grid` or `random`".format(search_type))

    val_score_grid = np.zeros((len(param_list), k))
  
    for i, (layers, opt_fn) in enumerate(param_list):
        logger.info("Model #{} of {}".format(i+1, len(param_list)))  
        for j, (train_idxs, val_idxs) in enumerate(StratifiedKFold(n_splits=k, shuffle=True).split(X, y)):
            logger.debug("Fold #{} of {}".format(j, k))
            X_train, X_val = X[train_idxs], X[val_idxs]
            y_train, y_val = y[train_idxs], y[val_idxs]
            
            bnn = bnn_init(layers=layers, optimiser_fn=opt_fn, **init_args)
            bnn.fit(X_train, y_train, **fit_args)
            val_score_grid[i,j] = bnn.score(X_val, y_val, **score_args)
    
    # Refit the best model on the whole dataset
    best_model_idx = val_score_grid.mean(axis=1).argmax()
    logger.info("Model {} had the largest mean validation metric: {:.3f} pm {:.3f} (mean pm std over {} folds)".format(
        best_model_idx,
        val_score_grid.mean(axis=1)[best_model_idx],
        val_score_grid.std(axis=1)[best_model_idx],
        k))
    bnn_best = bnn_init(layers=param_list[best_model_idx][0], optimiser_fn=param_list[best_model_idx][1], **init_args)
    bnn_best.fit(X, y, **fit_args)

    def get_layer_string(layer):
        layer_name = layer.__class__.__name__
        if hasattr(layer, 'units'):
            layer_name += "_{}".format(layer.units)
        return layer_name

    model_keys = [[get_layer_string(l) for l in layers] + [opt_fn().__class__.__name__] for layers, opt_fn in param_list]
    model_keys = [", ".join(key) for key in model_keys]
    print(model_keys)

    val_score_df = pd.DataFrame(np.hstack([np.array(model_keys)[:,np.newaxis], val_score_grid]))
    val_score_df.columns = ['model'] + ['val_score_{}'.format(j) for j in range(val_score_grid.shape[1])]
    
    return bnn_best, val_score_df