import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score

from tensorflow.compat.v1.losses import mean_squared_error
from tensorflow.nn import sigmoid_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2
from tensorflow.keras.optimizers import Adam

from .utils import make_1d2d

import logging
logger = logging.getLogger(__name__)

#
# TODO: return samples from predict methods - done but not debugged
# TODO: replace fit checks with a decorator
#

def default_layers(p, C):
	"""The default model layers - single 128-unit dense layer, batch normalization then a 
	Bayesian DenseLocalReparameterization final layer.

	If layers=[] in the constructor the network has this architecture
	"""
	layers = []
	layers.append(tf.keras.layers.Dense(128, activation='relu', input_shape=(p,)))
	layers.append(tf.keras.layers.BatchNormalization())
	layers.append(tfp.layers.DenseLocalReparameterization(C))
	return layers

TARGET_TYPES = {
	'BnnBinaryClassifier' : 'binary',
	'BnnScalarRegressor' : 'continuous'
}

class BnnBase(BaseEstimator, metaclass=ABCMeta):
	"""Bayesian neural network base class.

	This is an abstract class, so do not instantiate it directly - use BnnBinaryClassifier or BnnScalarRegressor
	"""

	@abstractmethod
	def __init__(self, layers, optimiser_fn, metrics, n_mc_samples, verbose):
		"""

		Args:
			layers: a list of network layers, the last of which should be a DenseLocalReparameterization layer
			optimizer_fn: callable tensorflor optimizer
			metrics: keras materics passed to keras method Sequential.compile - currently not fully supported
					so control the metric using the argument to the .score function
			n_mc_samples: the number of Monte Carlo samples used in any predictin function. This is the value
					used by default but can be overriden using the n_mc_samples argument in (e.g.) predict
			verbose: how much to print (0: nothing, 1: progress bar when predicting/fitting). This is the default
					value but can be overridden in individual function calls.
		"""
		self.layers = layers
		self.optimiser_fn = optimiser_fn
		self.metrics = metrics
		self.n_mc_samples = n_mc_samples
		self.verbose = verbose
		self.target_type = TARGET_TYPES[self.__class__.__name__]

		logger.debug("Constructed instance of {} with {} layers".format(
			self.__class__.__name__,
			len(layers)))

	def _build_model(self, X, y):
		"""Compile the keras model for the logits
		"""
		logging.debug("Fitting {} model with X shape {} and y shape {}".format(X.shape, y.shape, self.__class__.__name__))

		if len(self.layers) == 0:
			logging.debug("No layers defined - using default architecture for {}".format(self.__class__.__name__))
			self.layers = self._get_layers(X, y)
		self._logit_model = tf.keras.Sequential()
		for l in self.layers[:-1]:
			logger.debug("Adding layer {} to {}".format(l, self.__class__.__name__))
			self._logit_model.add(l.__class__.from_config(l.get_config())) # Deep copy of layers - doesn't work with DenseLocalReparameterization layer for some reason
		logger.debug("Adding DenseLocalReparameterization layer with {} units".format(self.layers[-1].units))
		self._logit_model.add(tfp.layers.DenseLocalReparameterization(self.layers[-1].units)) # Hack for now - will always use mean-field approximation
		self.p = self._logit_model.layers[0].input_shape[1]
		self.C = self._logit_model.layers[-1].units
		logger.debug("Compiling logit model for {}...".format(self.__class__.__name__))
		self._logit_model.compile(loss=self._elbo(), optimizer=self.optimiser_fn(), metrics=self.metrics)
		self._hmodel = tf.keras.Model(self._logit_model.input , self._logit_model.layers[-2].output)

		self.metrics_names = self._logit_model.metrics_names # To make sklearn cross-validation work - deprecated - will remove
		logger.debug("{} logit model was compiled with the following metrics: {}".format(self.__class__.__name__, self.metrics_names))

	def fit(self, X, y, **kwargs):
		"""Train the model on examples X and labels y

		Args:
			X: array of examples with shape (n_examples, n_variables)
			y: array of labels with shape (n_examples, 1)
			**kwargs: passed to the Keras fit method. See Keras docs for more details.

		Returns:
			The fitted model. Also stores the keras fit history as a class attribute.
		"""
		logger.debug("Fitting model with X shape {} and y shape {}".format(X.shape, y.shape))
		if X.ndim == 1:
			X = make_1d2d(X)
		try:
			getattr(self, "_logit_model")
		except AttributeError:
			logger.debug("Model has not been built yet")
			self._build_model(X, y)
			logger.debug("Model built")
		if X.shape[1] != self.p:
			raise ValueError("Model expects {} input dimensions, not {}".format(self.p, X.shape[1]))
		self._check_labels(y)
		# TODO: check optimizer here
		verbosity, kwargs = self._check_verbosity(kwargs)
		self.fit_history = self._logit_model.fit(X, y, verbose=verbosity, **kwargs)
		return self

	@abstractmethod
	def _get_layers(self, X, y):
		pass

	def loss(self, X, y, n_mc_samples=None, mean_only=True, **kwargs):
		"""Return the loss evaluated on examples X and labels y

		Args:
			X: examples (array with shape (n_examples, n_variables))
			y: labels (array with shape (n_examples, n_output_classes))
			n_mc_samples: number of Monte Carlo samples to use when evaluatingg the loss.
						Default is None, which uses the number defined in the constructor
			mean_only: whether to return the average loss over the Monte Carlo samples or
					the samples values (default is True - return the mean only)
			**kwargs: passed to Sequential.evaluate. See Keras docs for more details.

		Returns:
			The loss - float if n_mc_smaples=True otherwise array with shape n_mc_samples
			of loss samples.

		# TODO: separate loss and loss_samples methods?
		"""
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logger.debug("Calculating the loss using {} MC samples".format(n_mc_samples))
		verbosity, kwargs = self._check_verbosity(kwargs)
		loss_samples = np.array(
			[self._logit_model.evaluate(X, y, verbose=verbosity, **kwargs)[0] for _ in range(n_mc_samples)])
		if mean_only:
			return loss_samples.mean(axis=0)
		else:
			return loss_samples

	# Could make this protected
	def H(self, X, **kwargs):
		"""The (deterministic) activation of the penultimate layer for a set of examples X.

		Args:
			X: input with shape (n_examples, n_input_dimensions)
			**kwargs: passed to the Keras fit method. See Keras docs for more details.


		Returns:
			H: activations of the penultimate network layer, an array with shape (n_examples, penultimate_layer_size)
		"""
		logger.debug("Calculating H for X with shape {}".format(X.shape))
		check_is_fitted(self, "_logit_model")
		verbosity, kwargs = self._check_verbosity(kwargs)
		return self._hmodel.predict(X, verbose=verbosity, **kwargs)

	def var_params(self):
		"""The variational parameters of the final layer weights (the bias is deterministic but is
		is included with the weight posteriors as they are needed together to calculate RATE).

		TODO: what happens to these shapes when variational posterior is not fully factorised?

		Returns:
			M_W: the means of the variational posterior, shape (penultimate layer size, final layer size)
			V_W: the variances of the variational posterior, shape (penultimate layer size, final layer size)
			b: deterministic bias of the final layer, shape (final layer size,)
		"""
		check_is_fitted(self, "_logit_model")
		W1_loc, W1_scale, b = [K.eval(self._logit_model.layers[-1].kernel_posterior.distribution.loc),
							   K.eval(self._logit_model.layers[-1].kernel_posterior.distribution.scale),
							   K.eval(self._logit_model.layers[-1].bias_posterior.distribution.loc)]
		return W1_loc, np.square(W1_scale), b

	# Should have the class index as the first index, so that we can use numpy broadcasting for the batch multiplication
	def logit_posterior(self, X):
		"""The means and covariance of the posterior over the logits. Calculated using the variational
		posterior over the final layer weights.

		Args:
			X: examples (array with shape (n_examples, n_variables))

		TODO: shapes may break for C > 1

		Returns:
			logit_posterior, a 2-tuple containing:
				1. an array containing the logit posterior mean, shape (n_classes, n_examples)
				2. an array containing the logit posterior covariance, shape (n_classes, n_examples, n_examples)
		"""
		logger.debug("Calculating the logit posterior for X with shape {}".format(X.shape))
		H_X = self.H(X)
		M_W, V_W, b = self.var_params()
		M_F = np.matmul(H_X, M_W) + b[np.newaxis,:]
		V_F = np.array([np.matmul(H_X*V_W[:,c], H_X.T) for c in range(self.C)])
		return M_F.T, V_F

	@abstractmethod
	def _nll(self, labels, logits):
		"""Negative log likelihood - the reconstruction term in the ELBO
		"""
		pass

	def _elbo(self):
		"""The evidence lower bound - sum of negative log likelihood and KL-divergence term
		"""
		return lambda y_true, logits: tf.reduce_sum(self._nll(y_true, logits)) + sum(self._logit_model.losses)/K.cast(K.shape(y_true)[0], "float32")

	@abstractmethod
	def predict_samples(self, X, n_mc_samples=None, **kwargs):
		pass

	def _check_labels(self, y):
		"""Check that labels match model type
		"""
		if type_of_target(y) != self.target_type:
			raise ValueError("Label type is {} but model expects {}".format(type_of_target(y), self.target_type))

	# def evaluate(self, X, y, **kwargs):
	# 	return self.score(X, y, **kwargs)
	# Maybe better not to use this method due to possible confusion between keras and sklearn APIs

	def _check_verbosity(self, kwargs):
		if 'verbose' in kwargs:
			verbosity = kwargs.pop('verbose')
			return verbosity, kwargs
		else:
			return self.verbose, kwargs


class BnnBinaryClassifier(BnnBase, ClassifierMixin):
	"""Bayesian neural network for binary classification
	"""

	def __init__(self, layers=[], optimiser_fn=Adam, metrics=["acc"], n_mc_samples=100, verbose=0):
		"""Constructs a Bnn Binary Classifier. See BnnBase __init__ for the meanings of the arguments.
		
		This function contains some sensible defaults, including for the network architecture (the layers
		argument) but it is better to specify your own architecture.
		"""
		super().__init__(
			layers=layers,
			optimiser_fn=optimiser_fn,
			metrics=metrics,
			n_mc_samples=n_mc_samples,
			verbose=verbose)

	def _nll(self, labels, logits):
		"""Negative log likelihood - the reconstruction term in the ELBO. It is a sigmoid cross entropy
		for binary classification.

		Returns:
			The negative cross entropy tensorflow op
		"""
		return sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

	def predict(self, X, n_mc_samples=None, **kwargs):
		"""Predicted class labels over Monte Carlo samples. The mean prediction probability over the samples is
		thresholded (threshold is 0.5) to give class labels (as opposed to thresholding the sampled predicitons and then using the 
		mode of the predicted labels).

		Args:
			X: examples (array with shape (n_examples, n_variables))
			n_mc_samples: number of Monte Carlo samples. Default (None) uses the number defined in __init__
			**kwargs: passed to keras predict(). See keras documentation for more info.

		Returns:
			Predicted class labels - an array with shape (n_examples, 1)	
		"""
		check_is_fitted(self, "_logit_model")
		if X.ndim == 1:
			X = make_1d2d(X)
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logger.debug("Predicting mean class labels with X shape {} and {} MC samples".format(X.shape, n_mc_samples))
		verbosity, kwargs = self._check_verbosity(kwargs)
		logit_preds = np.squeeze(np.array([self._logit_model.predict(X, verbose=verbosity, **kwargs) for _ in range(n_mc_samples)]))
		proba_preds = 1.0/(1.0+np.exp(-logit_preds))
		return (proba_preds.mean(axis=0) >= 0.5).astype(int)

	def predict_proba(self, X, n_mc_samples=None, **kwargs):
		"""Returns mean of predicted class probabilities over Monte Carlo samples.

		As predict but returns the predicted class probabilities
		"""
		logger.debug("Predicting mean class probabilities with X shape {} and {} MC samples".format(X.shape, n_mc_samples))
		verbosity, kwargs = self._check_verbosity(kwargs)
		proba_preds = self.predict_proba_samples(X, n_mc_samples, verbose=verbosity, **kwargs)
		return proba_preds.mean(axis=0)

	def _get_layers(self, X, y):
		return default_layers(X.shape[1], 1)

	def predict_proba_samples(self, X, n_mc_samples=None, **kwargs):
		"""Return samples of predicted class probabilities for examples X.

		As predict_proba but the raw samples are returned, not just the mean
		"""
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logger.debug("Sampling predicted class probabilities with X shape {} and {} MC samples".format(X.shape, n_mc_samples))
		verbosity, kwargs = self._check_verbosity(kwargs)
		logit_preds = np.squeeze(np.array([self._logit_model.predict(X, verbose=verbosity, **kwargs) for _ in range(n_mc_samples)]))
		return 1.0/(1.0+np.exp(-logit_preds))

	def predict_samples(self, X, n_mc_samples=None, **kwargs):
		"""Return sampled class labels for exampels X. The sampled predictions are thresholded (at 0.5) to give labels.,
		which are returned.

		Args:
			X: examples, array with shape (n_examples, n_varialbes)
			n_mc_samples: number of Monte Carlo samples. Default (None) uses the number defined in __init__
			**kwargs: passed to keras predict(). See keras documentation for more info.			

		Returns:
			Sampled class labels, array with shape (n_examples, n_mc_samples)
		"""
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logger.debug("Sampling predicted class labels with X shape {} and {} MC samples".format(X.shape, n_mc_samples))
		verbosity, kwargs = self._check_verbosity(kwargs)
		logit_preds = np.squeeze(np.array([self._logit_model.predict(X, verbose=verbosity, **kwargs) for _ in range(n_mc_samples)]))
		proba_preds = 1.0/(1.0+np.exp(-logit_preds))
		return (proba_preds > 0.5).astype(int)

	def score(self, X, y, metric="accuracy", n_mc_samples=None, predict_args={}, **kwargs):
		"""Score the model on its predictions of labels y from examples X. The scoring metric is controlled by the metric argument.

		The two supported metrics (accuracy and area under ROC curve) use the sklearn.metric functions. **kwargs are
		passed to these functions to control (e.g.) sample weight etc

		Args:
			X: array of examples with shape (n_examples, n_variables)
			y: array of binary labels with shape (n_examples, 1)
			metric: the scoring metric. Default is "accuracy", but can also use "auc" (for area under ROC curve) or any callable
					function that takes arguments f(labels, prediction_probabilities, **kwargs)
			n_mc_samples: number of Monte Carlo samples. Default (None) uses the number defined in __init__
			predict_args: keyword-value pairs passed to predict/predict_proba.
			**kwargs: keyword-value pairs passed to metric.

		Returns:
			score as float
		"""
		if type_of_target(y) != self.target_type:
			raise ValueError("Label type is {} but model expects {}".format(type_of_target(y), self.target_type))
		if metric=="accuracy":
			return accuracy_score(y, self.predict(X, n_mc_samples, **predict_args), **kwargs)
		elif metric=="auc":
			return roc_auc_score(y, self.predict_proba(X, n_mc_samples, **predict_args), **kwargs)
		elif callable(metric):
			return metric(y, self.predict_proba(X, n_mc_samples, **predict_args), **kwargs)
		else:
			raise ValueError("metric must be either 'accuracy', 'auc' or callable")

class BnnScalarRegressor(BnnBase, RegressorMixin):
	"""Bayesian neural network for scalar regression
	"""

	def __init__(self, layers=[], optimiser_fn=Adam, metrics=["mse"], n_mc_samples=100, verbose=0):
		super().__init__(
			layers=layers,
			optimiser_fn=optimiser_fn,
			metrics=metrics,
			n_mc_samples=n_mc_samples,
			verbose=verbose)

	def _nll(self, labels, logits):
		"""Negative log likelihood - the reconstruction term in the ELBO
		"""
		return mean_squared_error(labels=labels, predictions=logits)

	def predict(self, X, n_mc_samples=None, **kwargs):
		"""Returns mean prediction over Monte Carlo samples
		"""
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logger.debug("Predicting mean response with X shape {} and {} MC samples".format(X.shape, n_mc_samples))
		verbosity, kwargs = self._check_verbosity(kwargs)
		return np.mean([self._logit_model.predict(X, verbose=verbosity, **kwargs) for _ in range(n_mc_samples)], axis=0)

	def predict_samples(self, X, n_mc_samples=None, **kwargs):
		check_is_fitted(self, "_logit_model")
		if n_mc_samples is None: n_mc_samples = self.n_mc_samples
		logger.debug("Sampling predicted response with X shape {} and {} MC samples".format(X.shape, n_mc_samples))
		verbosity, kwargs = self._check_verbosity(kwargs)
		return np.array([self._logit_model.predict(X, verbose=verbosity, **kwargs) for _ in range(n_mc_samples)])

	def _get_layers(self, X, y):
		return default_layers(X.shape[1], 1)