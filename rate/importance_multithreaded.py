# multithreaded code here

# put KLD computations in single function

import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Manager
import ray
import time

from .models import BnnBase, BnnBinaryClassifier
from .projections import CovarianceProjection
from .logutils import TqdmLoggingHandler

import logging
logger = logging.getLogger(__name__)
logger.addHandler(TqdmLoggingHandler())

#############################################
# 			Using multithreading			#
#############################################

# From https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html









#############################################
# 				Using ray					#
#############################################

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