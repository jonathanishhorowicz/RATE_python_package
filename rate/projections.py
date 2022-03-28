import numpy as np
from sklearn.linear_model import RidgeCV

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
		raise NotImplementedError

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

	def __repr__(self):
		return "covariance_projection"

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

		logger.debug("Calculating ESA posterior using the p-inv projection. Input shapes: X : {}, M_F : {} and V_F : {}".format(X.shape, M_F.shape, V_F.shape))

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

	def __repr__(self):
		return "pinv_projection"


class RidgeProjection(ProjectionBase):
	"""The pseudoinverse projection with an L2 penalty.
	
	lambda = 1/(2*alpha) where alpha is the inverse penalty used in 
	sklearn. This is selected using LOOCV.
	
	Attributes
		- alphas: sequence of alpha values
	"""
	def __init__(self, alphas=np.logspace(-5, 5, 11), ridgeCV_args={}):
		super().__init__()
		self.alphas = alphas
		self.model = None
		self.lambdas = []
		self.ridgeCV_args = {}
		self.cv_values = []
		
	def esa_posterior(self, X, M_F, V_F):
		"""Calculate the mean and covariance of the effect size analogue posterior.

		Args:
			X: array of inputs with shape (n_examples, n_variables)
			M_F: array of logit posterior means with shape (n_classes, n_examples)
			V_F: array of logit posterior covariances with shape (n_classes, n_examples, n_examples)
			**kwargs: passed to RidgeCV.fit
		
		Returns:
			effect_size_analogue_posterior: a 2-tuple containing:
				1. array of posterior means with shape (n_variables, n_classes)
				2. array of posterior covariances with shape (n_classes, n_variables, n_variables)
		"""
		
		if M_F.shape[0] != V_F.shape[0]:
			raise ValueError("Number of classes must match ({}!={})".format(M_F.shape[0], V_F.shape[0]))
		
		# fit a model per target to select penalty
		logger.debug("Calculating esa posterior using RidgeProjection")
		logger.debug("Input shapes: X: {}\tM_F: {}\tRidgeProjection_F:{}".format(
			X.shape, M_F.shape, V_F.shape))
		logger.debug("Fitting {} ridge estimators to find lambda".format(
			M_F.shape[0]))
		logger.debug("Trying {} alphas from {} to {}".format(
			len(self.alphas), np.amin(self.alphas), np.amax(self.alphas)))

		# center logits before fitting ridge model
		
		self.model = [
			RidgeCV(alphas=self.alphas, fit_intercept=False, **self.ridgeCV_args).fit(X, M_F[c])
					for c in range(M_F.shape[0])
		]
				
		self.lambdas = [1.0/(2.0*m.alpha_) for m in self.model]
		logger.debug("Selected lambdas: {}".format(self.lambdas))
		
		# Calculate effect size analogue posterior mean and covariance
		logger.debug("Calculating regularised pseudoinverses")
		X_pinvs = self._solve_reg_pinv(X)
		
		logger.debug("Calculating esa posterior mean")
		M_B = [np.matmul(X_pinvs[c], M_F[c,:,np.newaxis]).squeeze(axis=1) for c in range(len(X_pinvs))]
		
		logger.debug("Calculating esa posterior covariance")
		V_B = [np.matmul(np.matmul(X_pinvs[c], V_F[c]), X_pinvs[c].T) for c in range(len(X_pinvs))]
		
		return np.array(M_B), np.array(V_B)
		
	def _solve_reg_pinv(self, X):
		"""
		Returns solve(X' X + lam*I, X') for each lambda (one per class)

		lambda = 1/(2*alpha)
		"""
		return [
			np.linalg.solve(np.dot(X.T, X) + lam * np.identity(X.shape[1]), X.T)
			for lam in self.lambdas
		]

	def __repr__(self):
		return "ridge_projection"

# class PseudoinverseProjection(ProjectionBase):
#     """The pseudoinverse projection

#     Attributes:
#         - lam (float): regularisation strength
#     """
#     def __init__(self, lam=0.0):
#         super().__init__()
#         self._X_dagger = None
#         self.lam = lam

#     def esa_posterior(self, X, M_F, V_F):
#         """Calculate the mean and covariance of the effect size analogue posterior.

#         Args:
#             X: array of inputs with shape (n_examples, n_variables)
#             M_F: array of logit posterior means with shape (n_classes, n_examples)
#             V_F: array of logit posterior covariances with shape (n_classes, n_examples, n_examples)
		
#         Returns:
#             effect_size_analogue_posterior: a 2-tuple containing:
#                 1. array of posterior means with shape (n_variables, n_classes)
#                 2. array of posterior covariances with shape (n_classes, n_variables, n_variables)
#         """

#         logger.debug("Calculating ESA posterior using the p-inv projection. Input shapes: X : {}, M_F : {} and V_F : {}".format(X.shape, M_F.shape, V_F.shape))

#         # Check if X has been cached
#         if self._X is None:
#             logger.debug("No X is cached - calculating pinv(X) and storing it")
#             self._X = X
#             self._X_dagger = self._solve(self._X)
#         elif not np.array_equal(self._X, X):
#             logger.debug("Does not match cached X: calculating new pinv(X) and storing it")
#             self._X = X
#             self._X_dagger = self._solve(self._X) #np.linalg.pinv(self._X + self.lam*np.eye())

#         logger.debug("_X has shape {}, _X_dagger has shape {}".format(self._X.shape, self._X_dagger.shape))

#         # Calculate effect size analogue posterior mean and variance
#         M_B = np.matmul(self._X_dagger, M_F[:,:,np.newaxis])[:,:,0]
#         V_B = np.matmul(np.matmul(self._X_dagger, V_F), self._X_dagger.T)
		
#         logger.debug("Output shapes: M_B: {}, V_B: {}".format(M_B.shape, V_B.shape))

#         return M_B, V_B
	
#     def _solve(self, X):
#         return np.linalg.solve(np.dot(X.T, X) + self.lam * np.identity(X.shape[1]), X.T)
