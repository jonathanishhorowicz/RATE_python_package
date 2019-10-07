"""
Simulates binary classification datasets using latent functions with explicit latent functional forms
and then trains a neural network and Bayesian neural network.

Latent function is polynomial, which is thresholded to give binary labels. Threshold is random but 
chosen such that the class imbalance is never more sever than 60/40.

Variable importances are calculated using

1. RATE (Bayesian neural network)
2. Gradient-based methods, e.g. saliency map, integreated gradients etc... (neural network)
3. Gini importance of a tree ensemble mimic model (Bayesian neural network)
4. LASSO
"""
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV

from rate.models import BnnBinaryClassifier
from rate.projections import CovarianceProjection
from rate.mimic import train_mimic
from rate.importance import RATE2

import matplotlib.pyplot as plt
import seaborn as sns

from deepexplain.tensorflow import DeepExplain

import logging
logging.getLogger('rate').setLevel('INFO')

import pickle, time, timeit, inspect
import sys

start_time = time.ctime()

#
# Settings and initialisations
#
n_vals = [30000, 100000] # Dataset sizes
p_vals = [20, 50, 100, 300]
frac_p_causal_vals = [0.1, 0.3] # Fraction of variables that are causal
max_degree_vals = [3, 5]  # Maximum degree of latent polynomial
zero_prob_vals = [0.0] # Probability entries in X are zeroed
prob_flipy = 0.1 # Probability of flipping class label

test_size = 0.3
seed = 123
n_mc_samples = 10
n_repeats = 5 
effect_mask_prob = 0.7 # Probability of zeroing an effect size. Could just apply this to the non-linear terms

# Cross-validation settings
n_search_iter = 10 # Number of models tested in random search. Total number of fits is this*k
k = 3 # Numer of folds
n_jobs = int(sys.argv[1])

# Random forest CV grid
def rf_param_grid(p):
    return {
        'n_estimators' : np.arange(10, 1000, 10),
        'max_depth' : np.arange(1, p),
        'max_features' : ['auto', 'log2', 'sqrt']
    }

# Gradient boosting machine CV grid
def gbm_param_grid(p):
    return {
        'learning_rate' : [10.0**x for x in [-3.0, -2.0, -1.0]],
        'subsample' : [0.5, 0.7, 0.9, 1.0],
        'n_estimators' : np.arange(10, 1000, 10),
        'max_depth' : np.arange(1, p),
        'max_features' : ['auto', 'log2', 'sqrt']
    }

# Neural network training settings - same for BNN and NN
nn_train_args = {
    'epochs' : 30,
    'validation_split' : 0.3,
    'callbacks' : [EarlyStopping(monitor="val_acc", patience=3)]
}

def initialise_dict_of_lists():
    return {
        (n, p, frac_p_causal, max_degree, zero_prob) : []
            for n in n_vals
            for p in p_vals
            for frac_p_causal in frac_p_causal_vals
            for max_degree in max_degree_vals
            for zero_prob in zero_prob_vals
    }

# Simulation metadata
out = {}
out["metadata"] = {}
out["metadata"]["n_vals"] = n_vals
out["metadata"]["p_vals"] = p_vals
out["metadata"]["max_degree_vals"] = max_degree_vals
out["metadata"]["zero_prob_vals"] = zero_prob_vals
out["metadata"]["test_size"] = test_size
out["metadata"]["seed"] = seed
out["metadata"]["n_mc_samples"] = n_mc_samples
out["metadata"]["n_repeats"] = n_repeats
out["metadata"]["effect_mask_prob"] = effect_mask_prob
out["metadata"]["n_search_iter"] = n_search_iter
out["metadata"]["k"] = k
out["metadata"]["n_jobs"] = n_jobs
out["metadata"]["rf_param_grid"] = inspect.getsource(rf_param_grid)
out["metadata"]["gbm_param_grid"] = inspect.getsource(gbm_param_grid)
#out["metadata"]["nn_train_args"] =  nn_train_args
out["metadata"]["polynomial_powers"] = initialise_dict_of_lists()
out["metadata"]["effect_sizes"] = initialise_dict_of_lists()

# To store results
test_scores = pd.DataFrame()
variable_importances = pd.DataFrame()
timings = pd.DataFrame()

# Add names to each simulation setting and return as a dictionary
def key_to_df_dict(key):
    return {k : v for k, v in zip(['n', 'p', 'frac_p_causal', 'max_degree', 'zero_prob'], key)}

# Utility functions to store results
def add_test_score(model_name, score, dict_key, repeat_idx):
    return pd.concat([
        test_scores,
        pd.DataFrame({
            "model" : [model_name],
            "test accuracy" : [score],
            "repeat_idx" : [repeat_idx],
            **key_to_df_dict(dict_key)
        })],
        axis=0)

def add_importance_scores(method_name, importance_vals, dict_key, repeat_idx):
    p_causal = int(dict_key[1]*dict_key[2])
    return pd.concat([
        variable_importances,
        pd.DataFrame({
                "variable" : np.arange(dict_key[1]),
                "value" : importance_vals,
                "method" : method_name,
                "is_causal" : [True for _ in range(p_causal)] + [False for _ in range(p-p_causal)],
                "repeat_idx" : repeat_idx,
                **key_to_df_dict(dict_key)
            })],
        axis=0)

def add_timing(method_name, time, dict_key, repeat_idx):
    return pd.concat([
        timings,
        pd.DataFrame({
                "method" : [method_name],
                "time" : [time],
                "repeat_idx" : repeat_idx,
                **key_to_df_dict(dict_key)
            })],
        axis=0)

# Get an equivalent NN from a BNN - a bit hacky 
def get_deterministic_nn(bnn):
    """Returns a deterministic neural network with the same architecture as 
    a given Bayesian neural network. Binary classification only!"""
    if not isinstance(bnn, BnnBinaryClassifier):
        raise ValueError("get_deterministic_nn is only for BnnBinaryClassifiers")
    nn = Sequential()
    for layer in bnn._logit_model.layers[:-1]:
        deepcopied_layer = layer.__class__.from_config(layer.get_config())
        if hasattr(deepcopied_layer, 'kernel_initializer'):
            deepcopied_layer.build(layer.input_shape)
            deepcopied_layer.kernel.initializer.run(session=K.get_session()) # Reset the weights - very important
        nn.add(deepcopied_layer)
    nn.add(Dense(bnn._logit_model.layers[-1].output_shape[1]))
    nn.add(tf.keras.layers.Activation(activation="sigmoid"))
    nn.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return nn

# Add noise by flipping labels with probability prob_flipy
def flipy(y, prob_flipy):
    flipped_idxs = np.random.choice([True, False], replace=True, size=y.shape[0], p=[prob_flipy, 1.0-prob_flipy])
    y[flipped_idxs] = 1 - y[flipped_idxs]
    return y

for n in n_vals:
    for p in p_vals:
        for frac_p_causal in frac_p_causal_vals:
            p_causal = int(p*frac_p_causal)
            p_decoy = p-p_causal
            for max_degree in max_degree_vals:
                for zero_prob in zero_prob_vals:
                    dict_key = (n, p, frac_p_causal, max_degree, zero_prob)
                    print(key_to_df_dict(dict_key))
                    for repeat_idx in range(n_repeats):
                        print("repeat #{}".format(repeat_idx))
                        
                        #
                        # Simulate the data and labels
                        #
                        K.clear_session()
                        X = np.random.randn(n, p_causal)
                        mask = np.random.choice(2, replace=True, size=(n,p_causal), p=[zero_prob, 1.0-zero_prob])
                        X *= mask
                        
                        pf = PolynomialFeatures(3).fit(X) # First column is for the bias
                        X_poly = pf.transform(X)
                        effect_sizes = np.random.randn(X_poly.shape[1])
                        effect_size_mask = np.random.choice(2, size=X_poly.shape[1], replace=True, p=[effect_mask_prob, 1.0-effect_mask_prob])
                        effect_sizes *= effect_size_mask
                        f = np.dot(X_poly, effect_sizes)
                        y = (f > np.quantile(f, np.random.uniform(0.4, 0.6))).astype(int)[:,np.newaxis]
                        y = flipy(y, prob_flipy)


                        out["metadata"]["polynomial_powers"][dict_key].append(pf.powers_[effect_size_mask!=0,:])
                        out["metadata"]["effect_sizes"][dict_key].append(effect_sizes[effect_size_mask!=0])

                        # Add decoy variables
                        X = np.hstack([X, np.random.randn(n, p_decoy)])

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

                        print("Number of class 1: {},\tClass 2:{}".format(y.shape[0]-np.sum(y), np.sum(y)))

                        #
                        # Fit the models and evaluate their test accuracy
                        #

                        # BNN
                        bnn = BnnBinaryClassifier(verbose=0, n_mc_samples=n_mc_samples)
                        bnn.fit(X_train, y_train, **nn_train_args)
                        test_scores = add_test_score("Bayesian neural network", bnn.score(X_test, y_test), dict_key, repeat_idx)

                        nn = get_deterministic_nn(bnn)
                        nn.fit(X_train, y_train, verbose=0, **nn_train_args)
                        test_scores = add_test_score("neural network", nn.evaluate(X_test, y_test)[1], dict_key, repeat_idx)

                        #
                        # Variable importance
                        #

                        # LASSO
                        lasso_starttime = time.time()
                        logreg_l1 = LogisticRegressionCV(
                            Cs=10,
                            cv=k,
                            penalty="l1",
                            solver="saga",
                            n_jobs=n_jobs,
                            random_state=seed).fit(X_train, y_train)
                        test_scores = add_test_score("LASSO", logreg_l1.score(X_test, y_test), dict_key, repeat_idx)
                        variable_importances = add_importance_scores("LASSO", logreg_l1.coef_[0], dict_key, repeat_idx)
                        timings = add_timing("LASSO", time.time()-lasso_starttime, dict_key, repeat_idx)

                        # Mimic models
                        bnn_soft_predictions = bnn.predict_proba(X_train)
                        rf_mimic, rf_mimic_time = train_mimic(
                            RandomizedSearchCV(
                                RandomForestRegressor(),
                                rf_param_grid(p),
                                n_iter=n_search_iter,
                                cv=k,
                                n_jobs=n_jobs),
                            bnn, X_train, bnn_soft_predictions, X_test, n_mc_samples, True
                        )
                        variable_importances = add_importance_scores(
                            "random forest mimic", rf_mimic.best_estimator_.feature_importances_, dict_key, repeat_idx)
                        timings = add_timing("random forest mimic", rf_mimic_time, dict_key, repeat_idx)

                        gbm_mimic, gbm_mimic_time = train_mimic(
                            RandomizedSearchCV(
                                GradientBoostingRegressor(),
                                gbm_param_grid(p),
                                n_iter=n_search_iter,
                                cv=k,
                                n_jobs=n_jobs),
                            bnn, X_train, bnn_soft_predictions, X_test, n_mc_samples, True
                        )
                        variable_importances = add_importance_scores(
                            "gradient boosting machine mimic", gbm_mimic.best_estimator_.feature_importances_, dict_key, repeat_idx)
                        timings = add_timing("gradient boosting machine mimic", gbm_mimic_time, dict_key, repeat_idx)

                        # RATE
                        M_F, V_F = bnn.logit_posterior(X_test)
                        rate_vals, rate_time = RATE2(X_test, M_F, V_F, return_time=True)
                        variable_importances = add_importance_scores("RATE", rate_vals, dict_key, repeat_idx)
                        timings = add_timing("RATE", rate_time, dict_key, repeat_idx)

                        # Saliency-style maps, averaged over examples to give global importance. Average is of absolute value
                        with DeepExplain(session=K.get_session()) as de:
                            input_tensor = nn.layers[0].input
                            target_tensor = Model(inputs=input_tensor, outputs=nn.layers[-2].output)(input_tensor)

                            for attr_method in ["grad*input", "saliency", "intgrad", "elrp", "occlusion", "shapley_sampling"]:
                                arg_dict = {'samples' : 10} if "attr_method"=="shapley_sampling" else {}
                                s_time = time.time()
                                imp_vals = de.explain(
                                    attr_method,
                                    target_tensor, input_tensor,
                                    X_test, **arg_dict, ys=y_test,
                                    batch_size=64)
                                timings = add_timing(attr_method, time.time()-s_time, dict_key, repeat_idx)
                                variable_importances = add_importance_scores(
                                    attr_method, np.abs(imp_vals).mean(axis=0), dict_key, repeat_idx)
                                
                        # Save everything!
                        out["variable_importances"] = variable_importances
                        out["timings"] = timings
                        out["test_scores"] = test_scores
                        
                        with open("../numpy_objects/simulation_studies/out_n_vals_{}_pvals_{}_{}.pkl".format(n_vals, p_vals, start_time), "wb") as f:
                            pickle.dump(out, f)