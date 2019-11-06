import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Manager
import ray
import time
import rfpimp as rfp
from tqdm import tqdm

from .models import BnnBase, BnnBinaryClassifier
from .projections import CovarianceProjection
from .logutils import TqdmLoggingHandler

import logging
logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())

# TODO: make n_jobs/n_workers consistent across all of the code

@ray.remote
def KLD_ray(M_B_c, Lambda, j, nullify, exact):
	p = Lambda.shape[0]
	if nullify is not None:
		j = np.array(np.unique(np.concatenate(([j], nullify)), axis=0))
	m = M_B_c[j]
	Lambda_red = np.delete(Lambda, j, axis=0)[:,j]

	alpha = np.matmul(
		Lambda_red.T, 
		np.linalg.lstsq(
			np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
			Lambda_red,
			rcond=None)[0])

	# Approximation to the full KLD (equation S6 in AoAs supplemental)
	if nullify is None:
		kld = 0.5 * m**2.0 * alpha
	else:
		kld = 0.5 * np.matmul(np.matmul(m.T, alpha), m)

	# Additional terms in the full KLD calculation (equation 9 in AoAS paper)
	if exact:
		sigma_lambda_product = np.matmul(
						np.delete(np.delete(V_B[c], j, axis=0), j, axis=1),
						np.delete(np.delete(Lambda, j, axis=0), j, axis=1))

		kld += 0.5 * (
			- np.log(np.linalg.det(sigma_lambda_product) + 1e-9)
			+ np.trace(sigma_lambda_product)
			+ 1.0 - p)

	return kld

def RATE_ray(X, M_F, V_F, projection=CovarianceProjection(), nullify=None, 
	exact_KLD=False, jitter=1e-9, return_time=False, return_KLDs=False,
	n_jobs=1):

	if not (X.shape[0] == M_F.shape[1] == V_F.shape[1] == V_F.shape[2]):
		raise ValueError("Inconsistent number of examples across X and logit posterior")
	if M_F.shape[0] != V_F.shape[0]:
		raise ValueError("Inconsistent number of classes between logit posterior mean and covariance")

	# logger.info("Calculating RATE (using ray) values for {} classes, {} examples and {} variables and {} jobs".format(M_F.shape[0], X.shape[0], X.shape[1], n_jobs))
	# logger.debug("Input shapes: X: {}, M_F: {}, V_F: {}".format(X.shape, M_F.shape, V_F.shape))

	M_B, V_B = projection.esa_posterior(X, M_F, V_F)

	C = M_F.shape[0]
	p = X.shape[1]
	J = np.arange(p)
	if nullify is not None:
		J = np.delete(J, nullify, axis=0)

	KLDs = [np.zeros(J.shape[0]) for _ in range(C)]
	ray.init(num_cpus=n_jobs)

	start_time = time.time()
	for c in range(C):
		logger.info("Calculating RATE values for class {} of {}".format(c+1, C))
		Lambda = np.linalg.pinv(V_B[c] + jitter*np.eye(V_B.shape[1]))
		Lambda_id = ray.put(Lambda)
		KLDs[c] = ray.get([KLD_ray.remote(M_B[c], Lambda_id, j, nullify, exact_KLD) for j in J])
	ray.shutdown()

	if (np.array(KLDs) < 0.0).any():
		logger.warning("Some KLD values are negative - try a larger jitter value (current value: {})".format(jitter))
                                                 
	out = [klds / np.sum(klds) for klds in KLDs]
	rate_time = time.time() - start_time

	logger.info("The RATE calculation took {} seconds".format(round(rate_time, 3)))

	if C==1:
		out = out[0]
	
	if return_KLDs:
		out = [out, KLDs]
	if return_time:
		out = [out, rate_time]
	return out

def RATE2(*args, **kwargs):
	"""Wrapper for RATE - RATE2 was the development function name.

	For supporting older notebooks, scripts etc"""
	logger.warning("RATE2 is deprecated - please use rate")
	return RATE(*args, **kwargs)

def rate(X, M_F, V_F, projection=CovarianceProjection(), nullify=None, 
	exact_KLD=False, method="KLD", jitter=1e-9, return_time=False, return_KLDs=False,
	n_jobs=1, parallel_backend=""):
	"""Calculate RATE values. This function will replace previous versions in v1.0

	Args:
		X: array containing input data, shape (n_examples, n_variables)
		M_F: array containing logit posterior mean, shape (n_classes, n_examples).
		V_F: array containing logit posterior covariance, shape (n_classes, n_examples, n_examples).
		projection: an projection defining the effect size analogue. Must inherit from ProjectionBase. These are defined in projections.py
		nullify: array-like containing indices of variables for which RATE will not be calculated. Default `None`, in which case RATE values are calculated for every variable.
		exact_KLD: whether to include the log determinant, trace and 1-p terms in the KLD calculation. Default is False.
		method: Used in development. Use "KLD" (default) for the RATE calculation.
		jitter: added to the diagonal of the effect size analogue posterior to ensure positive semi-definitiveness. The code will warn you if any of the resulting KLD values
				are negative, in which case you should try a larger jitter. This is due to the covariance matrices of the logit posterior not being positive semi-definite.
		return_time: whether or not to return the time taken to compute the RATE values. Default if False.
		return KLDs: whether to return the KLD values as well as the RATE values. For debugging. Default is False.
		parallel_backend: the parallel backend (only relevant if n_jobs > 1). One of 'ray' or 'multiprocessing'
	
	Returns:
		rate_vals: a list of length n_classes, where each item is an array of per-variable RATE values for a given class. A single array is returned for n_classes = 1.
		If return_time=True then a 2-tuple containing rate_vals and the computation time is returned.
		If return_KLDs=True then the first item of the 2-tuple is itself a 2-tuple of (RATE_values, KLD_values)
	"""

	logger.debug("Input shapes: X: {}, M_F: {}, V_F: {}".format(X.shape, M_F.shape, V_F.shape))
	logger.debug("Using {} method".format(method))

	#
	# Shape checks. 1D M_F and 2D V_F will have extra dimension added at the front (for the output class)
	#
	if M_F.ndim==1 :
		M_F = M_F[np.newaxis]
		logger.debug("Reshaping 1D M_F to {}".format(M_F.shape))

	if V_F.ndim==2:
		V_F = V_F[np.newaxis]
		logger.debug("Reshaping 2D V_F to {}".format(V_F.shape))

	if not (X.shape[0] == M_F.shape[1] == V_F.shape[1] == V_F.shape[2]):
		raise ValueError("Inconsistent number of examples across X and logit posterior")
	if M_F.shape[0] != V_F.shape[0]:
		raise ValueError("Inconsistent number of classes between logit posterior mean and covariance")

	logger.info("Calculating RATE values for {} classes, {} examples and {} variables".format(M_F.shape[0], X.shape[0], X.shape[1]))

	# PARALLELISATION NOT FULLY TESTED YET - CALL RATE_ray directly
	# if n_jobs > 1:
	# 	if parallel_backend not in ["ray", "multiprocessing"]:
	# 		raise ValueError("{} is not a recognised parallel backend - choose `ray` or `multiprocessing`")
	# 	logger.info("Using {} parallel backend with {} jobs".format(parallel_backend, n_jobs))

	M_B, V_B = projection.esa_posterior(X, M_F, V_F)

	C = M_F.shape[0]
	p = X.shape[1]
	J = np.arange(p)
	if nullify is not None:
		J = np.delete(J, nullify, axis=0)

	KLDs = [np.zeros(J.shape[0]) for _ in range(C)]

	start_time = time.time()
	for c in range(C):
		logger.info("Calculating RATE values for class {} of {}".format(c+1, C))
		Lambda = np.linalg.pinv(V_B[c] + jitter*np.eye(V_B.shape[1]))
		for j in tqdm(J):
			if method=="KLD":
				if nullify is not None:
					j = np.array(np.unique(np.concatenate(([j], nullify)), axis=0))
				m = M_B[c,j]
				Lambda_red = np.delete(Lambda, j, axis=0)[:,j]

				alpha = np.matmul(
					Lambda_red.T, 
					np.linalg.lstsq(
						np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
						Lambda_red,
						rcond=None)[0])

				# Approximation to the full KLD (equation S6 in AoAs supplemental)
				if nullify is None:
					KLDs[c][j] = 0.5 * m**2.0 * alpha
				else:
					KLDs[c][j] = 0.5 * np.matmul(np.matmul(m.T, alpha), m)

				# Additional terms in the full KLD calculation (equation 9 in AoAS paper)
				if exact_KLD:
					sigma_lambda_product = np.matmul(
									np.delete(np.delete(V_B[c], j, axis=0), j, axis=1),
									np.delete(np.delete(Lambda, j, axis=0), j, axis=1)
									)

					KLDs[c][j] += 0.5 * (
						- np.log(np.linalg.det(sigma_lambda_product) + 1e-9)
						+ np.trace(sigma_lambda_product)
						+ 1.0 - p)

			elif method=="cond_var_red":
				Sigma = V_B[c]
				m = M_B[c,j]
				Sigma_red = np.delete(Sigma, j, axis=0)[:,j]

				KLDs[c][j] = np.matmul(
					Sigma_red.T, 
					np.linalg.lstsq(
						np.delete(np.delete(Sigma, j, axis=0), j, axis=1),
						Sigma_red,
						rcond=None)[0])

			elif method=="MI":
				Sigma = V_B[c]
				m = M_B[c,j]
				Sigma_red = np.delete(Sigma, j, axis=0)[:,j]

				alpha = np.matmul(
					Sigma_red.T, 
					np.linalg.lstsq(
						np.delete(np.delete(Sigma, j, axis=0), j, axis=1),
						Sigma_red,
						rcond=None)[0])
				KLDs[c][j] = -0.5 * np.log(1.0 - alpha/Sigma[j,j])

	logger.debug("{} of the KLD values are negative and {} of them are nan".format(np.sum(np.array(KLDs)<0.0), np.isnan(KLDs).sum()))

	if (np.array(KLDs) < 0.0).any():
		logger.warning("Some KLD values are negative - try a larger jitter value (current value: {})".format(jitter))
                                                 
	out = [klds / np.sum(klds) for klds in KLDs]
	rate_time = time.time() - start_time

	logger.info("The RATE calculation took {} seconds".format(round(rate_time, 3)))

	if C==1:
		out = out[0]
		KLDs = KLDs[0]

	if return_KLDs:
		out = [out, KLDs]
	if return_time:
		out = [out, rate_time]
	return out

def perm_importances(model, X, y, features=None, n_examples=None, n_mc_samples=100):
	"""
	Calculate permutation importances for a BNN or its mimic. Also returns the time taken
    so result is a 2-tuple (array of importance values, time)

	Args:
		model: a BnnBinaryClassifier, RandomForestClassifier or GradientBoostingClassifier
		X, y: examples and labels. The permutation importances are computed by shuffling columns
			  of X and seeing how the prediction accuracy for y is affected
		features: How many features to compute importances for. Default (None) is to compute
				  for every feature. Otherwise use a list of integers
		n_examples: How many examples to use in the computation. Default (None) uses all the
					features. Otherwise choose a positive integer that is less than 
					the number of rows of X/y.
		n_mc_samples: number of MC samples (BNN only)

	Returns a 1D array of permutation importance values in the same order as the columns of X
	"""
	X_df, y_df = pd.DataFrame(X), pd.DataFrame(y)
	X_df.columns = X_df.columns.map(str) # rfpimp doesn't like integer column names

	if n_examples is None:
		n_examples = -1
	start_time = time.time()
	if isinstance(model, BnnBinaryClassifier):
		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df,
								metric=lambda model, X, y, sw: model.score(X, y, n_mc_samples, sample_weight=sw), n_samples=n_examples, sort=False).values)
	elif isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df, n_samples=n_examples, sort=False).values)
	time_taken = time.time() - start_time
	return imp_vals, time_taken