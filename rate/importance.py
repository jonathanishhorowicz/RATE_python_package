import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Manager
#import ray
import time
#import rfpimp as rfp
from tqdm import tqdm

from scipy.linalg import solve_triangular, sqrtm

from .models import BnnBase, BnnBinaryClassifier
from .projections import CovarianceProjection
from .logutils import TqdmLoggingHandler

import logging
logger = logging.getLogger(__name__)
#logger.addHandler(TqdmLoggingHandler())

def qr_solve(A, b):
    Q, R = np.linalg.qr(A)
    return np.matmul(solve_triangular(R, Q.T), b)

def rate(X, M_F, V_F, projection=CovarianceProjection(), nullify=None, 
	exact_KLD=False, method="KLD", solver="qr", jitter=1e-9, return_time=False, return_KLDs=False,
	n_jobs=1, parallel_backend="", progressbar=False):
	"""Calculate RATE values. This function will replace previous versions in v1.0

	Args:
		X: array containing input data, shape (n_examples, n_variables)
		M_F: array containing logit posterior mean, shape (n_classes, n_examples).
		V_F: array containing logit posterior covariance, shape (n_classes, n_examples, n_examples).
		projection: an projection defining the effect size analogue. Must inherit from ProjectionBase. These are defined in projections.py
		nullify: array-like containing indices of variables for which RATE will not be calculated. Default `None`, in which case RATE values are calculated for every variable.
		exact_KLD: whether to include the log determinant, trace and 1-p terms in the KLD calculation. Default is False.
		method: Used in development. Use "KLD" (default) for the RATE calculation.
		solver: If 'qr', solve the linear system using QR (default). Choose 'lstsq' for a least-squares solution
		jitter: added to the diagonal of the effect size analogue posterior to ensure positive semi-definitiveness. The code will warn you if any of the resulting KLD values
				are negative, in which case you should try a larger jitter. This is due to the covariance matrices of the logit posterior not being positive semi-definite.
		return_time: whether or not to return the time taken to compute the RATE values. Default if False.
		return KLDs: whether to return the KLD values as well as the RATE values. For debugging. Default is False.
		parallel_backend: the parallel backend (only relevant if n_jobs > 1). One of 'ray' or 'multiprocessing'
		progressbar: whether to display the tqpdm progress bar (default False).
	
	Returns:
		rate_vals: a list of length n_classes, where each item is an array of per-variable RATE values for a given class. A single array is returned for n_classes = 1.
		If return_time=True then a 2-tuple containing rate_vals and the computation time is returned.
		If return_KLDs=True then the first item of the 2-tuple is itself a 2-tuple of (RATE_values, KLD_values)
	"""

	logger.debug("Input shapes: X: {}, M_F: {}, V_F: {}".format(X.shape, M_F.shape, V_F.shape))
	logger.debug("Using {} method and {} solver".format(method, solver))

	#
	# Shape checks. 1D M_F and 2D V_F will have extra dimension added at the front (for the output class)
	#
	if M_F.ndim==1:
		M_F = M_F[np.newaxis]
		logger.debug("Reshaping M_F to {}".format(M_F.shape))

	if V_F.ndim==2:
		V_F = V_F[np.newaxis]
		logger.debug("Reshaping 2D V_F to {}".format(V_F.shape))

	if not (X.shape[0] == M_F.shape[1] == V_F.shape[1] == V_F.shape[2]):
		raise ValueError("Inconsistent number of examples across X and logit posterior")
	if M_F.shape[0] != V_F.shape[0]:
		raise ValueError("Inconsistent number of classes between logit posterior mean and covariance")

	logger.info("Calculating RATE values for {} classes, {} examples and {} variables".format(
		M_F.shape[0], X.shape[0], X.shape[1]))
	if exact_KLD:
		logger.info("Using exact KLD calcuation")

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

	if solver == "qr":
		alpha_solve_fn = qr_solve
	elif solver == "lstsq":
		alpha_solve_fn = lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0]
	else:
		logger.warning("Unrecognised solver {}, using qr".format(solver))
		alpha_solve_fn = qr_solve

	start_time = time.time()
	for c in range(C):
		logger.info("Calculating RATE values for class {} of {}".format(c+1, C))
		Lambda = np.linalg.pinv(V_B[c] + jitter*np.eye(V_B.shape[1]))
		for j in tqdm(J, disable=not progressbar):
			if method=="KLD":
				if nullify is not None:
					j = np.array(np.unique(np.concatenate(([j], nullify)), axis=0))
				m = M_B[c,j]
				Lambda_red = np.delete(Lambda, j, axis=0)[:,j]

				alpha = np.matmul(
					Lambda_red.T, 
					alpha_solve_fn(
						np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
						Lambda_red))

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


def jth_partition(mu, Sigma, j):
    mu_j = np.array(mu[j]).reshape(1,1)
    mu_min_j = np.delete(mu, j, axis=0)[:,np.newaxis]
    sigma_j = np.array(Sigma[j,j]).reshape(1,1)
    sigma_min_j = np.delete(Sigma, j, axis=0)[:,j][:,np.newaxis]
    Sigma_min_j = np.delete(np.delete(Sigma, j, axis=0), j, axis=1)
    
#     print("Sizes:\n\tmu_j: {}, mu_min_j: {}\n\tsigma_j: {}, sigma_min_j:{}, Sigma_min_j:{}".format(
#         mu_j.shape, mu_min_j.shape, sigma_j.shape, sigma_min_j.shape, Sigma_min_j.shape
#     ))
    
    return mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j

def condition_gaussian(mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j):
    mu_cond = mu_min_j - np.dot(sigma_min_j, mu_j)/sigma_j
    #print("\n\tmu_cond: {}".format(mu_cond.shape))
    Sigma_cond = Sigma_min_j - np.dot(sigma_min_j, np.dot(mu_j, sigma_min_j.T))/sigma_j
    #print("\tSigma_cond: {}".format(Sigma_cond.shape))
    
    return mu_cond, Sigma_cond

def Wasserstein_gaussian(mu_0, Sigma_0, mu_1, Sigma_1):
    """https://github.com/VersElectronics/WGPOT/blob/master/wgpot.py"""

    sqrtK_0 = sqrtm(Sigma_0)
    first_term = np.dot(sqrtK_0, Sigma_1)
    K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

    cov_dist = np.trace(Sigma_0) + np.trace(Sigma_1) - 2.0 * np.trace(sqrtm(K_0_K_1_K_0))
    l2norm = np.sum(np.square(np.abs(mu_0 - mu_1)))
    d = np.real(np.sqrt(l2norm + cov_dist))

    return d

def rate_wasserstein(X, M_F, V_F, projection=CovarianceProjection()):
    M_B, V_B = projection.esa_posterior(X, M_F, V_F)
    C = M_F.shape[0]
    
    wass_unnorm = [np.zeros(X.shape[1]) for _ in range(M_F.shape[0])]
    
    for c in range(C):
        logger.info("Calculating Wasserstein RATE values for class {} of {}".format(c+1, C))
        for j in range(X.shape[1]):
            mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(M_B[c], V_B[c], j)
            mu_cond, Sigma_cond = condition_gaussian(mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j)
            wass_unnorm[c][j] = Wasserstein_gaussian(mu_cond, Sigma_cond, mu_min_j, Sigma_min_j)
    
    return [wass/wass.sum() for wass in wass_unnorm]

# def perm_importances(model, X, y, features=None, n_examples=None, n_mc_samples=100):
# 	"""
# 	Calculate permutation importances for a BNN or its mimic. Also returns the time taken
#     so result is a 2-tuple (array of importance values, time)

# 	Args:
# 		model: a BnnBinaryClassifier, RandomForestClassifier or GradientBoostingClassifier
# 		X, y: examples and labels. The permutation importances are computed by shuffling columns
# 			  of X and seeing how the prediction accuracy for y is affected
# 		features: How many features to compute importances for. Default (None) is to compute
# 				  for every feature. Otherwise use a list of integers
# 		n_examples: How many examples to use in the computation. Default (None) uses all the
# 					features. Otherwise choose a positive integer that is less than 
# 					the number of rows of X/y.
# 		n_mc_samples: number of MC samples (BNN only)

# 	Returns a 1D array of permutation importance values in the same order as the columns of X
# 	"""
# 	X_df, y_df = pd.DataFrame(X), pd.DataFrame(y)
# 	X_df.columns = X_df.columns.map(str) # rfpimp doesn't like integer column names

# 	if n_examples is None:
# 		n_examples = -1
# 	start_time = time.time()
# 	if isinstance(model, BnnBinaryClassifier):
# 		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df,
# 								metric=lambda model, X, y, sw: model.score(X, y, n_mc_samples, sample_weight=sw), n_samples=n_examples, sort=False).values)
# 	elif isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
# 		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df, n_samples=n_examples, sort=False).values)
# 	time_taken = time.time() - start_time
# 	return imp_vals, time_taken

def vanilla_gradients(model, X, numpy=True):
    """Computes the vanilla gradients of model output w.r.t inputs.

    Args:
        model: keras model
        X: input array
        numpy: True to return numpy array, otherwise returns Tensor

    Returns:
        Gradients of the predictions w.r.t input
    """
    X_tensor = tf.cast(X, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor)

    grads = tape.batch_jacobian(preds, X_tensor)
    if numpy:
        grads = grads.numpy()
    return grads

def gradient_input(model, X, numpy=True):
    """Computes the gradients*inputs, where gradients are of model
    output wrt input

    Args:
        model: keras model
        X: input array
        numpy: True to return numpy array, otherwise returns Tensor

    Returns:
        Gradients of the predictions w.r.t input
    """
    gradients = vanilla_gradients(model, X, False)
    gradients_inputs = tf.math.multiply(gradients, X[:,tf.newaxis,:])
    if numpy:
        gradients_inputs = gradients_inputs.numpy()
    return gradients_inputs

def integrated_gradients(model, X, n_steps=20, numpy=True):
    """Integrated gradients using zero baseline
    
    https://keras.io/examples/vision/integrated_gradients/
    
    Args:
        model: keras model
        X: input array
        n_steps: number of interpolation steps
        numpy: True to return numpy array, otherwise returns Tensor
        
    Returns:
        Integrated gradients wrt input
    """
    
    baseline = np.zeros(X.shape).astype(np.float32)
        
    # 1. Do interpolation.
    X = X.astype(np.float32)
    interpolated_X = [
        baseline + (step / n_steps) * (X - baseline)
        for step in range(n_steps + 1)
    ]
    interpolated_X = np.array(interpolated_X).astype(np.float32)
    
    # 2. Get the gradients
    grads = []
    for i, x in enumerate(interpolated_X):
        grad = vanilla_gradients(model, x)
        grads.append(grad)
    
    # 3. Approximate the integral using the trapezoidal rule
    grads = np.array(grads)
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = grads.mean(axis=0)
    
    # 4. Calculate integrated gradients and return
    integrated_grads = (X - baseline)[:,np.newaxis,:] * avg_grads
    
    return integrated_grads

def smoothed_gradients(model, X, noise=1.0, n_samples=10, numpy=True):
    """SmoothGrad
    
    Args:
        model: keras model
        X: input array
        noise: variance of Gaussian noise added to each pixel
        n_samples: number of noisy samples
        numpy: True to return numpy array, otherwise returns Tensor
        
    Returns:
        SmoothGrad wrt input
    """
    X = X.astype(np.float32)
    
    # 1. Add noise then get the gradients
    noisy_grads = []
    for i in range(n_samples):
        noisy_grad = vanilla_gradients(model, X + np.random.normal(0.0, noise, X.shape))
        noisy_grads.append(noisy_grad)
    noisy_grads = tf.convert_to_tensor(noisy_grads, dtype=tf.float32)
    
    # 2. Mean noisy gradient
    avg_noisy_grads = tf.reduce_mean(noisy_grads, axis=0)
    
    if numpy:
        avg_noisy_grads = avg_noisy_grads.numpy()
    return avg_noisy_grads

def guided_backprop(model, X, numpy=True):
    preds = model(X)[:,:,tf.newaxis]
    grads = vanilla_gradients(model, X, False)
    
    guided_grads = (
                tf.cast(preds > 0, "float32")
                * tf.cast(preds > 0, "float32")
                * grads
            )
    if numpy:
        guided_grads = guided_grads.numpy()
    return guided_grads