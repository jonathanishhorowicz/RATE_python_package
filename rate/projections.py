import numpy as np

from abc import ABCMeta, abstractmethod

import logging
logger = logging.getLogger(__name__)

# TODO: add functions that use samples rather than closed-form

class ProjectionBase(metaclass=ABCMeta):
	"""Base class for projections used to calculate effect size analogues
	"""
	@abstractmethod
	def __init__(self):
		self._X = None # Cached data matrix

	@abstractmethod
	def esa_posterior(self, X, M_F, V_F):
		pass

class CovarianceProjection(ProjectionBase):
	"""The covariance projection
	"""
	def __init__(self):
		super().__init__()
		self._X_c = None # Cached centred data matrix

	def esa_posterior(self, X, M_F, V_F):
		"""Calculate the mean and covariance of the effect size analogue posterior.

		Args:
			X: array of inputs with shape (n_examples, n_variables)
			M_F: array of logit posterior means with shape (n_classes, n_examples)
			V_F: array of logit posterior covariances with shape (n_classes, n_examples, n_examples)
		
		Returns:
			effect_size_analogue_posterior: a 2-tuple containing:
				1. array of posterior means with shape (n_classes, n_variables)
				2. array of posterior covariances with shape (n_classes, n_variables, n_variables)
		"""

		# Check if X has been cached
		logger.debug("Calculating ESA posterior using the covariance projection. Input shapes: X : {}, M_F : {} and V_F : {}".format(X.shape, M_F.shape, V_F.shape))
		if self._X is None:
			logger.debug("No X is cached - calculating centered X and storing it")
			self._X = X
			self._X_c = X - X.mean(axis=0, keepdims=True)
		elif not np.array_equal(self._X, X):
			logger.debug("Does not match cached X: calculating new centered X and storing it")
			self._X = X
			self._X_c = X - X.mean(axis=0, keepdims=True)

		n = self._X.shape[0]
		logger.debug("n = {}".format(n))
		M_F_c = M_F - M_F.mean(axis=1, keepdims=True)
		logger.debug("M_F_c has shape {}, self._X has shape {}, self._X_c has shape {}".format(M_F_c.shape, self._X.shape, self._X_c.shape))
		M_B = 1.0/(n-1.0) * np.matmul(self._X_c.T, M_F_c[:,:,np.newaxis])[:,:,0]
		# M_B = 1.0/(n-1.0) * np.einsum('ij,kj -> ik', M_F_c) check if these two lines are equivalent
		V_B = 1.0/(n-1.0)**2.0 * np.matmul(np.matmul(self._X_c.T, V_F), self._X_c)

		logger.debug("Output shapes: M_B: {}, V_B: {}".format(M_B.shape, V_B.shape))

		return M_B, V_B

class PseudoinverseProjection(ProjectionBase):
	"""The pseudoinverse projection
	"""
	def __init__(self):
		super().__init__()
		self._X_dagger = None

	def esa_posterior(self, X, M_F, V_F):
		"""Calculate the mean and covariance of the effect size analogue posterior.

		Args:
			X: array of inputs with shape (n_examples, n_variables)
			M_F: array of logit posterior means with shape (n_classes, n_examples)
			V_F: array of logit posterior covariances with shape (n_classes, n_examples, n_examples)
		
		Returns:
			effect_size_analogue_posterior: a 2-tuple containing:
				1. array of posterior means with shape (n_variables, n_classes)
				2. array of posterior covariances with shape (n_classes, n_variables, n_variables)
		"""

		logger.debug("Calculating ESA posterior using the covariance projection. Input shapes: X : {}, M_F : {} and V_F : {}".format(X.shape, M_F.shape, V_F.shape))

		# Check if X has been cached
		if self._X is None:
			logger.debug("No X is cached - calculating pinv(X) and storing it")
			self._X = X
			self._X_dagger = np.linalg.pinv(self._X)
		elif not np.array_equal(self._X, X):
			logger.debug("Does not match cached X: calculating new pinv(X) and storing it")
			self._X = X
			self._X_dagger = np.linalg.pinv(self._X)

		logger.debug("_X has shape {}, _X_dagger has shape {}".format(self._X.shape, self._X_dagger.shape))

		# Calculate effect size analogue posterior mean and variance
		M_B = np.matmul(self._X_dagger, M_F[:,:,np.newaxis])[:,:,0]
		V_B = np.matmul(np.matmul(self._X_dagger, V_F), self._X_dagger.T)
		
		logger.debug("Output shapes: M_B: {}, V_B: {}".format(M_B.shape, V_B.shape))

		return M_B, V_B


