import numpy as np
import multiprocessing as mp
import time
import rfpimp as rfp
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

#from GPy.core import GP
from .models import BnnBase
from .projections import CovarianceProjection, PseudoinverseProjection	

# TODO: make n_jobs/n_workers consistent across all of the code

def RATE2(X, M_F, V_F, projection=CovarianceProjection(), nullify=None, method="KLD"):
	"""Calculate RATE values. This function will replace previous versions in v1.0

	Args:
		X: array containing input data, shape (n_examples, n_variables)
		M_F: array containing logit posterior mean, shape (n_classes, n_examples).
		V_F: array containing logit posterior covariance, shape (n_classes, n_examples, n_examples).
		projection: an projection defining the effect size analogue. Must inherit from ProjectionBase
		nullify: array-like containing indices of variables for which RATE will not be calculated. Default `None`, in which case RATE values are calculated for every variable.
	
	Returns:
		rate_vals: a list of length n_classes, where each item is an array of per-variable RATE values for a given class. A single array is returned for n_classes = 1.
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
	J = np.arange(X.shape[1])
	if nullify is not None:
		J = np.delete(np.arange(Lambda.shape[0]), nullify, axis=0)

	KLDs = [np.zeros(J.shape[0]) for _ in range(C)]

	for c in range(C):
		logger.info("Calculating RATE values for class {} of {}".format(c, C))
		Lambda = np.linalg.pinv(V_B[c])
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

				if nullify is None:
					KLDs[c][j] = 0.5 * m**2.0 * alpha
				else:
					KLDs[c][j] = 0.5 * np.matmul(np.matmul(m.T, alpha), m)
                                                 
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
                                                 
	out = [klds / np.sum(klds) for klds in KLDs]

	if C==1:
		return out[0]
	else:
		return out

# def get_esa_posterior(model, X, projection):
# 	if isinstance(model, BnnBase):
# 		C = model.C
# 		M_W, V_W, b = model.var_params()
# 		H = model.H(X)
# 		return esa_posterior_bnn(X, H, M_W, V_W, b, C, projection)
# 	elif isinstance(model, GP):
# 		return esa_posterior_gp(model, X)
# 	else:
# 		raise ValueError("Model type {} is not supported".format(type(model)))

# def esa_posterior_gp(gp, X, projection):
# 	"""Regression/binary classification only
# 	"""
# 	M_F, V_F = gp.predict_f_full_cov(X)
# 	if projection == "covariance":
# 		M_F -= M_F.mean(axis=0)[np.newaxis,:] # Centering
# 		X_c = X - X.mean(axis=0)[np.newaxis,:]
# 		M_B = 1.0/(n-1.0) * np.matmul(X_C.T, M_F)
# 		V_B = 1.0(n-1.0)**2.0 * np.matmul(np.matmul(X_c.T, V_F), X_c)
# 	elif projection == "linear":
# 		GenInv = np.linalg.pinv(X)
# 		M_B = np.matmul(GenInv, M_F)
# 		V_B = np.matmul(np.matmul(GenInv, V_F), GenInv.T)

# def esa_posterior_bnn(X, H, M_W, V_W, b, C, projection="covariance"):
# 	"""
# 	Compute the means and covariance of the effect size analogues B for a Bayesian neural network
# 	described in (Ish-Horowicz et al., 2019)

# 	Args:
# 		X: array of inputs with shape (n_examples, n_input_dimensions)
# 		H: array of penultimate network layer outputs with shape (n_examples, final_hidden_layer_size)
# 		M_W: Array of final weight matrix means with shape (final_hidden_layer_size, n_classes)
# 		V_W: Array of final weight matrix variances with shape (final_hidden_layer_size, final_hidden_layer_size, n_classes)
# 		b: Final layer (deterministic) bias
# 		C: number of classes
# 		projection: Projection operator for computing effect size analogues. Either "linear" (pseudoinverse) or "covariance" (default).

# 	Returns:
# 		M_B: an array of means of B, the multivariate effect size analgoues, with shape (n_classes, n_pixels)
# 		V_B: an array of covariance of B, the multivariate effect size analgoues, with shape (n_classes, n_pixels, n_pixels)

# 	"""
# 	assert X.shape[0]==H.shape[0], "Number of examples (first dimension) of X and H must match"
# 	assert b.shape[0]==C, "Number of bias units must match number of nodes in the logit layer"
# 	assert M_W.shape[1]==C, "Second dimension of logit weight means must match number of classes"
# 	assert V_W.shape[1]==C, "Second dimension of logit weight variances must match number of classes"
# 	assert M_W.shape[0]==V_W.shape[0], "means and variances of logit weight matrix must have the same shape"
# 	assert H.shape[1]==M_W.shape[0], "Second dimension of logit weight means must match H.shape[1], the penultimate layer size"

# 	n = H.shape[0]

# 	# Logits
# 	M_F = np.matmul(H, M_W) + b[np.newaxis,:]
# 	V_F = np.array([np.matmul(H*V_W[:,c], H.T) for c in range(C)])

# 	# Effect size analogues
# 	if projection == 'covariance':
# 		# Centred logits
# 		M_F_c = M_F - M_F.mean(axis=0)[np.newaxis,:]
# 		V_F_c = V_F # NB ignoring the additional variance due to centering + 1.0/n**2.0 * V_F.mean(axis=0)
# 		X_c = X - X.mean(axis=0)[np.newaxis,:]
# 		M_B = 1.0/(n-1.0) * np.array([np.matmul(X_c.T, M_F_c[:,c]) for c in range(C)])
# 		V_B = 1.0/(n-1.0)**2.0 * np.array([np.matmul(np.matmul(X_c.T, V_F_c[c,:,:]), X_c)for c in range(C)])
# 	elif projection == 'linear': 
# 		GenInv = np.linalg.pinv(X)
# 		M_B_mat = np.matmul(GenInv, M_F)
# 		M_B = np.array([M_B_mat[:, c] for c in range(C)]) # What does this line do?
# 		V_B = np.array([np.matmul(np.matmul(GenInv, V_F[c, :, :]), GenInv.T) for c in range(C)])
# 	else: 
# 		raise ValueError("Unrecognised projection {}, please use `covariance` or `linear`".format(projection))
	    
# 	return M_B, V_B
    
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

def perm_importances(model, X, y, features=None, n_examples=None, n_mc_samples=100):
	"""
	Calculate permutation importances for a BNN or its mimic. Also returns the time taken
    so result is a 2-tuple (array of importance values, time)

	Args:
		model: a BNN_Classifier, RandomForestClassifier or GradientBoostingClassifier
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
	if isinstance(model, BNN_Classifier):
		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df,
								metric=lambda model, X, y, sw: model.score(X, y, n_mc_samples, sample_weight=sw), n_samples=n_examples, sort=False).values)
	elif isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
		imp_vals = np.squeeze(rfp.importances(model, X_df, y_df, n_samples=n_examples, sort=False).values)
	time_taken = time.time() - start_time
	return imp_vals, time_taken