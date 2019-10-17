import numpy as np
import multiprocessing as mp
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
						np.delete(np.delete(Lambda, j, axis=0), j, axis=1)
						)

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

	logger.info("Calculating RATE (using ray) values for {} classes, {} examples and {} variables and {} jobs".format(M_F.shape[0], X.shape[0], X.shape[1], n_jobs))
	logger.debug("Input shapes: X: {}, M_F: {}, V_F: {}".format(X.shape, M_F.shape, V_F.shape))

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
	
	if return_time:
		out = [out, rate_time]
	if return_KLDs:
		out = [out, KLDs]
	return out

def RATE2(X, M_F, V_F, projection=CovarianceProjection(), nullify=None, 
	exact_KLD=False, method="KLD", jitter=1e-9, return_time=False, return_KLDs=False,
	n_jobs=1):
	"""Calculate RATE values. This function will replace previous versions in v1.0

	Args:
		X: array containing input data, shape (n_examples, n_variables)
		M_F: array containing logit posterior mean, shape (n_classes, n_examples).
		V_F: array containing logit posterior covariance, shape (n_classes, n_examples, n_examples).
		projection: an projection defining the effect size analogue. Must inherit from ProjectionBase. These are defined in projections.py
		nullify: array-like containing indices of variables for which RATE will not be calculated. Default `None`, in which case RATE values are calculated for every variable.
		exact_KLD: whether to include the log determinant, trace and 1-p terms in the KLD calculation. Default is False.
		method: from when I was investigating the effect of the bug. Just use "KLD" (default), which is the correct RATE calculation
		jitter: added to the diagonal of the effect size analogue posterior to ensure positive semi-definitiveness. The code will warn you if any of the resulting KLD values
				are negative, in which case you should try a larger jitter. This is due to the covariance matrices of the logit posterior not being positive semi-definite.
		return_time: whether or not to return the time taken to compute the RATE values. Default if False.
		return KLDs: whether to return the KLD values as well as the RATE values. For debugging. Default is False.
	
	Returns:
		rate_vals: a list of length n_classes, where each item is an array of per-variable RATE values for a given class. A single array is returned for n_classes = 1.
		If return_time=True then a 2-tuple containing rate_vals and the computation time is returned.
		If return_KLDs=True then the first item of the 2-tuple is itself a 2-tuple of (RATE_values, KLD_values)
	"""

	if not (X.shape[0] == M_F.shape[1] == V_F.shape[1] == V_F.shape[2]):
		raise ValueError("Inconsistent number of examples across X and logit posterior")
	if M_F.shape[0] != V_F.shape[0]:
		raise ValueError("Inconsistent number of classes between logit posterior mean and covariance")

	logger.info("Calculating RATE values for {} classes, {} examples and {} variables".format(M_F.shape[0], X.shape[0], X.shape[1]))
	logger.debug("Input shapes: X: {}, M_F: {}, V_F: {}".format(X.shape, M_F.shape, V_F.shape))
	logger.debug("Using {} method".format(method))

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
				if full_KLD:
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

	if (np.array(KLDs) < 0.0).any():
		logger.warning("Some KLD values are negative - try a larger jitter value (current value: {})".format(jitter))
                                                 
	out = [klds / np.sum(klds) for klds in KLDs]
	rate_time = time.time() - start_time

	logger.info("The RATE calculation took {} seconds".format(round(rate_time, 3)))

	if C==1:
		out = out[0]
	
	if return_time:
		out = [out, rate_time]
	if return_KLDs:
		out = [out, KLDs]
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

# ****************************************************************************************************
# ****************************************************************************************************
# ************************ DEPRECATED CODE BELOW THIS POINT - WILL BE REMOVED ************************
# ****************************************************************************************************
# ****************************************************************************************************
    
def RATE_sequential(mu_c, Lambda_c, nullify=None):
    """
	Compute RATE values fromt means and covariances of the effect
	size analgoues of a single class.

	Args:
		mu_c: Array of means of the effect size analgoues with shape (n_variables,).
		Lambda_c: Array of covariances of the effect size analogues with shape (n_variables, n_variables)
		nullify: Array of indices to be ignored (default None means include all variables).

	Returns:
		Array of RATE values with shape (n_variables,)
    """

    mu = mu_c
    Lambda = Lambda_c
    
    J = np.arange(Lambda.shape[0])
    if nullify is not None:
        J = np.delete(J, nullify, axis=0)

    print("Computing RATE with {} variables".format(J.shape[0]))
    print("Variable #:", end=" ")
    
    def single_marker_kld(j):
        
        if j%100==0:
            print(j, end=" ")
        
        if nullify is not None:
            j = np.array(np.unique(np.concatenate(([j], nullify)), axis=0))
        m = mu[j]
        Lambda_red = np.delete(Lambda, j, axis=0)[:,j]
        
        # in def'n of alpha below, 
        # changing np.linalg.solve to np.linalg.lstsq with rcond=None, to avoid singularity error 
        # when simulating data in Power Simulation Response to ICML Feedback.ipynb
        # (dana)
        alpha = np.matmul(Lambda_red.T, 
                          np.linalg.lstsq(np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
                                          Lambda_red, rcond=None)[0])
        if nullify is None:
            return 0.5 * m**2.0 * alpha
        else:
            return 0.5 * np.matmul(np.matmul(m.T, alpha), m)
    
    KLD = [single_marker_kld(j) for j in J]
    print("done")
    return KLD / np.sum(KLD)

def groupRATE_sequential(mu_c, Lambda_c, groups, nullify=None):
	"""
	Group RATE

	Args:
		groups: List of lists, where groups[i] contains the indices of the variables in group i
	"""
	mu = mu_c
	Lambda = Lambda_c
    
	J = np.arange(Lambda.shape[0])
	if nullify is not None:
		J = np.delete(J, nullify, axis=0)

	print("Computing RATE with {} groups".format(len(groups)))
	print("Group #:", end=" ")

	def group_kld(group, idx):
		if idx%100 == 0:
			print(idx, end=", ")

		if nullify is not None:
			j = np.array(np.unique(np.concatenate((group, nullify)), axis=0))
		else:
			j = group
		m = mu[j]
		Lambda_red = np.delete(Lambda, j, axis=0)[:,j]

		# in def'n of alpha below, 
		# changing np.linalg.solve to np.linalg.lstsq with rcond=None, to avoid singularity error 
		# when simulating data in Power Simulation Response to ICML Feedback.ipynb
		# (dana)
		alpha = np.matmul(Lambda_red.T, 
							np.linalg.lstsq(np.delete(np.delete(Lambda, j, axis=0), j, axis=1),
											Lambda_red, rcond=None)[0])
		if nullify is None:
			return 0.5 * m**2.0 * alpha
		else:
			return 0.5 * np.matmul(np.matmul(m.T, alpha), m)

	KLD = [group_kld(group, idx) for idx, group in enumerate(groups)]
	print("done")
	return KLD, KLD/np.sum(KLD)

# Worker initialisation and function for groupRATE
var_dict = {}

def init_worker(mu, Lambda, p):
	var_dict["mu"] = mu
	var_dict["Lambda"] = Lambda
	var_dict["p"] = p

def worker_func(worker_args):
	"""
	Returns KLD
	"""
	j, idx, filepath = worker_args
	Lambda_np = np.frombuffer(var_dict["Lambda"]).reshape(var_dict["p"], var_dict["p"])
	mu_np = np.frombuffer(var_dict["mu"])
	m = mu_np[j]
	Lambda_red = np.delete(Lambda_np, j, axis=0)[:,j]

	alpha = np.matmul(Lambda_red.T, 
					  np.linalg.lstsq(np.delete(np.delete(Lambda_np, j, axis=0), j, axis=1),
									  Lambda_red, rcond=None)[0])

	if isinstance(m, float):
		out = 0.5 * m**2.0 * alpha
	else:
		out = 0.5 * np.matmul(np.matmul(m.T, alpha), m)
	
	if filepath is not None:
		with open(filepath + "kld_{}.csv".format(idx), "w") as f:
			f.write(str(out))

	return out

def RATE(mu_c, Lambda_c, nullify=None, n_workers=1, filepath=None):
	if nullify is not None and n_workers > 1:
		logging.warning("Using nullify means a sequential RATE calculation")
		n_workers = 1
	
	if n_workers == 1:
		return RATE_sequential(mu_c, Lambda_c, nullify)
    
	p = mu_c.shape[0]
    
	print("Computing RATE for {} variables using {} worker(s)".format(p, n_workers))

	# Setup shared arrays
	mu_mp = mp.RawArray('d', p)    
	Lambda_mp = mp.RawArray('d', p*p)
	mu_np = np.frombuffer(mu_mp, dtype=np.float64)
	Lambda_np = np.frombuffer(Lambda_mp, dtype=np.float64).reshape(p, p)
	np.copyto(mu_np, mu_c)
	np.copyto(Lambda_np, Lambda_c)

    # Run pooled computation
	with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(mu_c, Lambda_c, p)) as pool:
		result = np.array(pool.map(worker_func, [(j, j, filepath) for j in range(p)]))
	return result/result.sum()

def groupRATE(mu_c, Lambda_c, groups, nullify=None, n_workers=1, filepath=None):
    if nullify is not None and n_workers > 1:
        logging.warning("Using nullify means a sequential groupRATE calculation")
        n_workers = 1

    if n_workers == 1:
        return groupRATE_sequential(mu_c, Lambda_c, groups, nullify)
    
    p = mu_c.shape[0]
    
    # Setup shared arrays
    mu_mp = mp.RawArray('d', p)    
    Lambda_mp = mp.RawArray('d', p*p)
    mu_np = np.frombuffer(mu_mp, dtype=np.float64)
    Lambda_np = np.frombuffer(Lambda_mp, dtype=np.float64).reshape(p, p)
    np.copyto(mu_np, mu_c)
    np.copyto(Lambda_np, Lambda_c)
        
    # Run pooled computation
    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(mu_c, Lambda_c, p)) as pool:
        result = np.array(pool.map(worker_func, [(group, idx, filepath) for idx, group in enumerate(groups)]))
    return result/result.sum()

# TODO: THIS RETURNS AN ARRAY WITH AN EXTRA AXIS IF C=1
def RATE_BNN(bnn, X, groups=None, nullify=None, effect_size_analogue="covariance",
	n_workers=1, return_esa_posterior=False, filepath=None):
	"""
	Compute RATE values for the Bayesian neural network described in (Ish-Horowicz et al., 2019).
	If C>2 there is one RATE value per pixel per class. Note that a binary classification task
	uses C=1 as there is a single output node in the network.

	This function wraps compute_B (which computes effect size analgoues) and RATE.

	Args:
		bnn: BNN object
		X: array of inputs with shape (n_examples, n_input_dimensions)
		groups: A list of lists where groups[i] is a list of indices of the variables in group i. Default is None (no group RATE)
		effect_size_analogue: Projection operator for computing effect size analogues. Either "linear" (pseudoinverse) or "covariance" (default).
		n_workers: number of workers for groupRATE
        return_esa_posterior: Controls whether the mean/covariance of the effect size analgoue posterior is also returned (default False)
        filepath: where to save the result of each worker (None means no saving and is the default)
		
	Returns:
		Tuple of (rate_vales, computation_time)
		If groups is None: rate_vales is a list of arrays of RATE values with length C. Each array in the list has shape (n_variables,).
		Otherwise, each array in rate_values has length len(groups).
	"""
	C = bnn.C
	M_W, V_W, b = bnn.var_params()
	H = bnn.H(X)

	start_time = time.time()

	M_B, V_B = compute_B(X, H, M_W, V_W, b, C, effect_size_analogue)

	try:
		if C > 2:
			if groups is None:
				out = [RATE(mu_c=M_B[c,:],Lambda_c=V_B[c,:,:], nullify=nullify, n_workers=n_workers, filepath=filepath) for c in range(C)]
			else:
				out = [groupRATE(mu_c=M_B[c,:],Lambda_c=V_B[c,:,:], groups=groups, nullify=nullify, n_workers=n_workers, filepath=filepath) for c in range(C)]
			rate_time = time.time() - start_time
		else:
			if groups is None:
				out = RATE(mu_c=M_B[0,:],Lambda_c=V_B[0,:,:], nullify=nullify, n_workers=n_workers, filepath=filepath)
			else:
				out = groupRATE(mu_c=M_B[0,:],Lambda_c=V_B[0,:,:], groups=groups, nullify=nullify, n_workers=n_workers, filepath=filepath)
			rate_time = time.time() - start_time
	except np.linalg.LinAlgError as err: # Redo this
		if 'Singular matrix' in str(err):
			logging.info("Computing RATE led to singlar matrices. Try using the nullify argument to ignore uninformative variables.")
			logging.error("Singular matrix", exc_info=True)
		else:
			raise err
            
	if return_esa_posterior:
		return out, rate_time, M_B, V_B
	else:
		return out, rate_time

