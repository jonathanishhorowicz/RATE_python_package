import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import is_regressor, is_classifier
from sklearn.utils.multiclass import type_of_target

import time
import logging
logger = logging.getLogger(__name__)

def mean_soft_prediction(bnn, X, n_mc_samples):
	"""
	Mean prediction by a BNN over MC samples. Predicted probabiltity if classification 
	and raw prediction in regression
	"""
	if is_classifier(bnn):
		return bnn.predict_proba(X, n_mc_samples)
	elif is_regressor(bnn):
		return bnn.predict(X, n_mc_samples)
	else:
		raise ValueError("BNN is neither regressor nor classifier!")


def train_mimic(mimic_model, bnn, x_train, y_train=None, x_test=None, n_mc_samples=100, return_time=False):
	"""
	Get the random forest trained to mimic the mean
	predictions of a Bayesian neural network. The mimic model is a regression model trained
	trained on the soft predictions (the logits) of the BNN.

	TODO: better to pass prediction lambda function as argument rather than the bnn

	Model selection is performed using random search cross-validation with 10 iterations and 5 folds - this can be quite
	slow but shouldn't take more than 10 minutes when parallelised over all available
	cores. Default behaviour is to use one core.
	
	Args:
		mimic_model: a Scikit-Learn model that implements the fit and score methods. Depending on the context
					 this could be a regression model (e.g. RandomForestRegressor) or a cross-validation search
					 object from sklearn.model_selection (e.g. RandomizedSearchCV).
		bnn_object: an instance of the BNN class. If `y_train` is not `None` then this argument is ignored when training the mimic
					model. The BNN is still used if `x_test` is provided.
		x_train: array of training examples with shape (n_examples, n_features).
					The random forest will be trained on these examples and their
					BNN predictions. The size of the second dimension must match the number of input dimensions expected by the BNN.
		y_train: array of soft labels which are used to train the mimic model. Default is `None`, in which case the labels are generated
					from `bnn`.
		x_test: array of test examples with shape (n_examples, n_features).
					If provided (default is None) then the random forest will be
					evaluated by comparing its predictions
					to those of the BNN. The size of the second dimension must match the number of input dimensions expected by the BNN
		n_mc_samples: the number of MC samples used when making BNN predictions.
						Their mean is used as the labels for the random forest.
		return_time: whether or not to return the time taken to compute the RATE values. Default if False.
	"""
	logger.debug("Fitting mimic model of type {} on inputs with shape {} and {} MC samples".format(type(mimic_model), x_train.shape, n_mc_samples))
	logger.debug("Supplied arguments: y_train: {}, x_test: {}".format(y_train is not None, x_test is not None))
	if isinstance(mimic_model, BaseSearchCV):
		if not is_regressor(mimic_model.estimator):
			raise ValueError("Mimic model must be a regression model")
	else:
		if not is_regressor(mimic_model):
			raise ValueError("Mimic model must be a regression model")

	if (bnn is not None) and (bnn.p != x_train.shape[1]):
		raise ValueError("Number of BNN input dimensions must match x_train")

	if x_test is not None:
		if x_train.shape[1] != x_test.shape[1]:
			raise ValueError("x_train and x_test must have the same number of dimensions")

	if y_train is not None:
		logger.warning("Using provided y_train and ignoring the supplied BNN for mimic training")
		if type_of_target(y_train) != "continuous":
			raise ValueError("Use soft labels (predicted probabilities) to train a mimic model")
	else:
		y_train = mean_soft_prediction(bnn, x_train, n_mc_samples)

	start_time = time.time()
	fit_result = mimic_model.fit(x_train, y_train)
	fit_time = time.time() - start_time

	if x_test is not None:
		mimic_test_r2 = fit_result.score(x_test, mean_soft_prediction(bnn, x_test, n_mc_samples))
		logger.info("Mimic R^2 on x_test: {:.3f}".format(mimic_test_r2))

	if return_time:
		return mimic_model, fit_time
	else:
		return mimic_model