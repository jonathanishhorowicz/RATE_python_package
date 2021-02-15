import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Process, Manager
import time

from scipy.linalg import solve_triangular, sqrtm

from .projections import CovarianceProjection

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

import logging
logger = logging.getLogger(__name__)

def qr_solve(A, b):
	"""Solve Ax=b for x using the QR decomposition
	"""
	Q, R = np.linalg.qr(A)
	return np.matmul(solve_triangular(R, Q.T), b)

def rate2(
	X,
	M_F, V_F,
	projection=CovarianceProjection(),
	nullify=[],
	excluded_vars=[],
	groups=None,
	solver="qr",
	jitter=1e-9,
	clip_KLDs=False):
	"""
	Calculate RATE values without inverting the entire covariance matrix at once.
	
	Will replace rate eventually. Slower than the old code as it stores a lot of
	intermediate results.
	
	Each term of the KLD is returned separately.
	
	Args:
		X: array containing input data, shape (n_examples, n_variables)
		M_F: array containing logit posterior mean, shape (n_classes, n_examples).
		V_F: array containing logit posterior covariance, shape (n_classes, n_examples, n_examples).
		projection: an projection defining the effect size analogue. Must inherit from ProjectionBase (see projections.py).
		nullify: list of indices of variables for which RATE will not be calculated.
				 Default is an empty list.
		excluded_vars: list of variable indices of variables that are totally excluded from the calculation.
						Not the same as nullify. Default is an empty list.
		groups: list of lists defining the variable groupings. Default is None (no groups).
		solver: If 'qr', solve the linear system using QR (default). Choose 'lstsq' for a least-squares solution.
		
	Returns:
		Length-2 list containing
			- list of dataframes of decomposed KLDs (one per class)
			- list of ranks of various intermediate matrices (one per class)
	"""
	
	# some input checks
	if len(nullify)>0 and len(excluded_vars)>0:
		intersection = np.intersect1d(nullify, excluded_vars)
		if intersection.shape[0]>0:
			raise ValueError("nullify and excluded_vars can't share elements: {}".format(intersection))
	
	if groups is not None:
		unique_grouped_vars = np.unique([item for sublist in groups for item in sublist])
		intersection = np.intersect1d(
			unique_grouped_vars,
			np.concatenate([nullify, excluded_vars])
		)
		if intersection.shape[0]>0:
			raise ValueError("Variables in groups can't be excluded/nullified: {}".format(intersection))

		vars_in_groups = np.isin(
			np.arange(X.shape[1]),
			np.concatenate(
				[unique_grouped_vars, excluded_vars, nullify],
				axis=0)
		)

		if not vars_in_groups.all():
			logger.warning("{} variables are not in a group, excluded or nullified".format(vars_in_groups.sum()))

	p_original = X.shape[1]

	if len(excluded_vars) > 0:
		recode_vars = True
		vars_to_keep = np.arange(X.shape[1])[~np.isin(np.arange(X.shape[1]), excluded_vars)]
		logger.debug("Removing {} variables. {} remain".format(
			len(excluded_vars), vars_to_keep.shape[0]
		))
		new2old_map = dict(zip(range(vars_to_keep.shape[0]), vars_to_keep))
		old2new_map = dict(zip(vars_to_keep, range(vars_to_keep.shape[0])))
		
		X = X[:,vars_to_keep]
		
		# recode input variable indices since some have been removed
		if len(nullify)>0:
			nullify = [old2new_map[idx] for idx in nullify]
		
		if groups is not None:
			groups = [[old2new_map[idx] for idx in g] for g in groups]
	else:
		recode_vars = False

	if len(nullify) > 0:
		logger.debug("{} variables are nullifed".format(len(nullify)))

	C = M_F.shape[0]
	p = X.shape[1]

	# effect size analogue posterior
	M_B, V_B = projection.esa_posterior(X, M_F, V_F)
	logger.debug("M_B shape is {}, V_B shape is {}".format(M_B.shape, V_B.shape))
	logger.info("V_B has rank {}".format(np.linalg.matrix_rank(V_B)))
	
	# symmetrise V_Bs
	for c,v in enumerate(V_B):
		if not np.array_equal(v, v.T):
			if not np.allclose(v, v.T):
				logger.warning("V_B[{}] may not be symmetric (max diff {})".format(
					c, "{:.3e}".format(np.amax(v-v.T))))
			else:
				logger.debug("Symmetrising V_B[{}]".format(c))
				v = 0.5*(v+v.T)

	# setup the variables to be iterated over
	if groups is None:
		if len(nullify)==0:
			J = [np.array([j]) for j in range(p)]
		else:
			J = [np.array([j]) for j in range(p) if j not in nullify]
	else:
		J = [np.array(g) for g in groups]
	
		if len(nullify)>0:
			J = [np.concatenate([j,nullify]) for j in J]
			
	#
	# store the terms in the KLD separately
	KLDs = [np.zeros((len(J), 4)) for _ in range(C)]
	cov_matrix_ranks = [np.zeros((len(J), 6)) for _ in range(C)]

	#
	# the solver for the linear system in the KLD term
	if solver == "qr":
		alpha_solve_fn = qr_solve
	elif solver == "lstsq":
		alpha_solve_fn = lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0]
	else:
		logger.warning("Unrecognised solver {}, using qr".format(solver))
		alpha_solve_fn = qr_solve

	# calculate rate values per class
	for c in range(C):
		logger.info("Calculating RATE values for class {} of {}".format(c+1, C))
		logger.debug("V_B[{}] has rank {}".format(c, np.linalg.matrix_rank(V_B[c])))

		for out_idx, j in enumerate(J):
							
			if len(nullify) > 0:
				j = np.array(np.unique(np.concatenate((j, nullify)), axis=0))
				
			logger.debug("at iteration {} j={} has shape {} and type {}".format(out_idx, j, j.shape, j.dtype))
			
			# partition mean and covariance
			mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(M_B[c], V_B[c], j)
			
			# conditional distribution
			(mu_cond, Sigma_cond), matrix_ranks_and_shapes = condition_gaussian(
				mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j
			)
			
			# store KLD results
			tmp = kl_mvn(
				mu_min_j, Sigma_min_j, mu_cond, Sigma_cond,
				jitter=jitter)

			KLDs[c][out_idx] = tmp[0]
			cov_matrix_ranks[c][out_idx] = matrix_ranks_and_shapes
	
	#   
	# format final result
	KLDs = [pd.DataFrame(klds, columns=["quad", "trace", "det", "KLD"]) for klds in KLDs]
	if groups is None:
		variable_column = np.array(J).squeeze(axis=1)
		if recode_vars:
			variable_column = [new2old_map[idx] for idx in variable_column]
		[KLDs[c].insert(0, "variable", variable_column) for c in range(C)]
	else:
		if recode_vars:
			groups = [[new2old_map[idx] for idx in g] for g in groups]
		[KLDs[c].insert(0, "group", ["group{}".format(x) for x in range(len(groups))]) for c in range(C)]
		[KLDs[c].insert(1, "variable", groups) for c in range(C)]
		KLDs = [kld.explode("variable") for kld in KLDs]
				
	#
	# add excluded/nullified indices
	if len(nullify) > 0:
		if recode_vars:
			nullify = [new2old_map[idx] for idx in nullify]
		extra_rows = make_extra_rows(nullify, "nullified", groups is not None)
		KLDs = [pd.concat([kld, extra_rows], axis=0) for kld in KLDs]
		
	if len(excluded_vars) > 0:
		extra_rows = make_extra_rows(excluded_vars, "excluded", groups is not None)
		KLDs = [pd.concat([kld, extra_rows], axis=0) for kld in KLDs]
		
	# variable indices should be ints all should be present in output
	for kld in KLDs:
		kld.variable = kld.variable.astype(int)
		kld.sort_values(by="variable", inplace=True)
		assert np.isin(np.arange(p_original), kld.variable.unique()).all()
		assert kld.shape[0]==p_original
		
	cov_matrix_ranks = [pd.DataFrame(
		x,
		columns=[
			"rank(Sigma_min_j)", "rank(Sigma_cond)", "rank(Sigma_update)",
			"ndim(Sigma_min_j)", "ndim(Sigma_cond)", "ndim(Sigma_update)"
		]
	) for x in cov_matrix_ranks]

	return [KLDs, cov_matrix_ranks]


def kl_mvn(m0, S0, m1, S1, jitter=0.0):
	"""
	Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
	Also computes KL divergence from a single Gaussian pm,pv to a set
	of Gaussians qm,qv.

	- accepts stacks of means, but only one S0 and S1
	- returns the three terms separately plus the total KLD in one list
	- also returns rank of S1 and S0 in a second list
	
	https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv

	From wikipedia
	KL( (m0, S0) || (m1, S1))
		 = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
				  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
	"""
	# store inv diag covariance of S1 and diff between means
	N = m0.shape[0]
	diff = m1 - m0
	
	if jitter > 0.0:
		S0 += jitter * np.identity(S0.shape[0])
		S1 += jitter * np.identity(S0.shape[0])

	# kl is made of three terms
	iS1S0 = np.linalg.solve(S1, S0)
	tr_term   = np.trace(iS1S0)
	det_term  = np.log((np.linalg.det(S1)/np.linalg.det(S0)) + 1e-9) 
	quad_term = diff.T @ np.linalg.solve(S1, diff) 

	logger.debug("quad_term: {} \t tr_term: {} \t det_term: {}".format(
		quad_term, tr_term, det_term
	))
	
	# kld = .5 * (tr_term + det_term + quad_term - N)
	
	return [
			[quad_term, tr_term, det_term, .5 * (tr_term + det_term + quad_term - N)],
			[np.linalg.matrix_rank(S0), np.linalg.matrix_rank(S1), np.linalg.matrix_rank(iS1S0)]
		   ]

def make_extra_rows(var_list, fill_value, include_groupcol):
	df = pd.concat(
		[
			pd.Series(var_list, index=range(len(var_list)), name="variable"),
			pd.DataFrame(np.nan, index=range(len(var_list)), columns=["quad", "trace", "det"]),
			pd.Series(fill_value, index=range(len(var_list)), name="KLD")
		],
		axis=1)

	if include_groupcol:
		df.insert(0, "group", ["no_group" for _ in range(df.shape[0])])
	return df

def jth_partition(mu, Sigma, j):
	"""
	Given mean vector mu and covariance matrix Sigma for the joint distribution over
	all p variables, partitions into the variable with index j and the remaining
	variables.

	j can be an array if multiple variables are included at once (GroupRATE)
	
	Args:
		- mu: p x 1 array, mean vector
		- Sigma: pxp array, covariance matrix
		- j: int, array of ints: variables to be conditioned on

	Returns:
		- mu_j: float or array of float, mean of p(\tilde{\beta}_{j})
		- mu_min_j:  array of floats, mean of p(\tilde{\beta}_{-j})
		- sigma_j, float or 2d array of floatS, (co)variance of p(\tilde{\beta}_{j})
		- sigma_min_j: array of floats, covariance(included vars, excluded vars)
		- Sigma_min_j: array of floats, covariance of p(\tilde{\beta}_{-j})
	"""
	
	logger.debug("j={}, mu has shape {}, Sigma has shape {}".format(j, mu.shape, Sigma.shape))
	
	mu_j = np.array(mu[j])[:,np.newaxis]
	mu_min_j = np.delete(mu, j, axis=0)[:,np.newaxis]
	sigma_j = Sigma[np.ix_(j,j)]
	sigma_min_j = np.delete(Sigma, j, axis=0)[:,j]
	Sigma_min_j = np.delete(np.delete(Sigma, j, axis=0), j, axis=1)
	
	logger.debug("Sizes:\n\tmu_j: {}, mu_min_j: {}\n\tsigma_j: {}, sigma_min_j:{}, Sigma_min_j:{}".format(
		mu_j.shape, mu_min_j.shape, sigma_j.shape, sigma_min_j.shape, Sigma_min_j.shape
	))
	
	return mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j


def condition_gaussian(mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j):
	"""
	Calculate parameters (mean, covariance) of conditional distribution over effect size
	analogues.
	
	This is the distribution p(\tilde{\beta}_{-j} | p(\tilde{\beta}_{j}=0) and calculated from
	the appropriate partioning of p(\tilde{\beta}).
	
	Returns two lists: mean/covariance in item 0 and the ranks/shapes of relevant matrices in item 1
	"""
	mu_cond = mu_min_j - np.dot(sigma_min_j, np.linalg.solve(sigma_j, mu_j))
	#print("\n\tmu_cond: {}".format(mu_cond.shape))
	
	# is this is a low-rank update? 
	Sigma_update = np.dot(sigma_min_j, np.linalg.solve(sigma_j, sigma_min_j.T))
	Sigma_cond = Sigma_min_j - Sigma_update
	
	matrix_ranks_and_shapes = [
		np.linalg.matrix_rank(Sigma_min_j),
		np.linalg.matrix_rank(Sigma_cond),
		np.linalg.matrix_rank(Sigma_update),
		Sigma_min_j.shape[0],
		Sigma_cond.shape[0],
		Sigma_update.shape[0],
	]
	
	return [[mu_cond, Sigma_cond], matrix_ranks_and_shapes]

def Wasserstein_gaussian(mu_0, Sigma_0, mu_1, Sigma_1):
	"""Wasserstein distance between two multivariate normals

	https://github.com/VersElectronics/WGPOT/blob/master/wgpot.py
	"""

	logger.debug("calcuating sqrtK_0 = sqrtm(Sigma_0)")
	sqrtK_0 = sqrtm(Sigma_0)

	logger.debug("calcuating np.dot(sqrtK_0, Sigma_1)")
	first_term = np.dot(sqrtK_0, Sigma_1)
	logger.debug("calcuating np.dot(first_term, sqrtK_0)")
	K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

	logger.debug("calcuating np.trace(Sigma_0) + np.trace(Sigma_1) - 2.0 * np.trace(sqrtm(K_0_K_1_K_0))")   
	cov_dist = np.trace(Sigma_0) + np.trace(Sigma_1) - 2.0 * np.trace(sqrtm(K_0_K_1_K_0))

	logger.debug("calcuating np.sum(np.square(np.abs(mu_0 - mu_1)))")
	l2norm = np.sum(np.square(np.abs(mu_0 - mu_1)))

	logger.debug("calcuating np.real(np.sqrt(l2norm + cov_dist))")
	d = np.real(np.sqrt(l2norm + cov_dist))


	return d

def rate_wasserstein(X, M_F, V_F, projection=CovarianceProjection()):

	M_B, V_B = projection.esa_posterior(X, M_F, V_F)
	C = M_F.shape[0]
	
	wass_unnorm = [np.zeros(X.shape[1]) for _ in range(M_F.shape[0])]
	
	for c in range(C):
		logger.info("Calculating Wasserstein RATE values for class {} of {}".format(c, C-1))
		for j in range(X.shape[1]):
			logger.debug("j={}".format(j))
			mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(M_B[c], V_B[c], j)
			logger.debug("partitioned esa posterior")
			mu_cond, Sigma_cond = condition_gaussian(mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j)
			logger.debug("conditioned gaussian")
			wass_unnorm[c][j] = Wasserstein_gaussian(mu_cond, Sigma_cond, mu_min_j, Sigma_min_j)
			logger.debug("calculated Wasserstein distance")
		logger.debug("For class {} there are:\n{} negative values, {} non-finite values".format(
			c, np.sum(wass_unnorm[c]<0.0), np.sum(~np.isfinite(wass_unnorm[c]))))

	if (np.array(wass_unnorm) < 0.0).any():
		logger.warning("Some rate values are negative")


	return [wass/wass.sum() for wass in wass_unnorm]


#
# this function is deprecated - contained too much old code
# use rate2 instead
#
def rate(X, M_F, V_F, projection=CovarianceProjection(), nullify=None, 
	exact_KLD=False, method="KLD", solver="qr", jitter=1e-9, return_time=False, return_KLDs=False,
	decompose_KLD=False, trim_KLDs=True, n_jobs=1):
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
	if decompose_KLD:
		logger.debug("Decomposing KLD")

	M_B, V_B = projection.esa_posterior(X, M_F, V_F)

	C = M_F.shape[0]
	p = X.shape[1]
	J = np.arange(p)
	if nullify is not None:
		J = np.delete(J, nullify, axis=0)

	KLDs = [np.zeros(J.shape[0]) for _ in range(C)]
	if decompose_KLD:
		if exact_KLD:
			KLDs_decomp = [np.zeros((J.shape[0],3)) for _ in range(C)] # alpha, mu^2, logdet term
		else:
			KLDs_decomp = [np.zeros((J.shape[0],2)) for _ in range(C)] # alpha, mu^2

	if solver == "qr":
		alpha_solve_fn = qr_solve
	elif solver == "lstsq":
		alpha_solve_fn = lambda A, b: np.linalg.lstsq(A, b, rcond=-1)[0]
	else:
		logger.warning("Unrecognised solver {}, using qr".format(solver))
		alpha_solve_fn = qr_solve

	start_time = time.time()
	for c in range(C):
		logger.info("Calculating RATE values for class {} of {}".format(c+1, C))
		Lambda = np.linalg.pinv(V_B[c] + jitter*np.eye(V_B.shape[1]))
		for j in J:
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

				if decompose_KLD:
					KLDs_decomp[c][j,0] = alpha
					KLDs_decomp[c][j,1] = m**2.0 if nullify else np.sum(m**2.0)


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

					logdet_term = 0.5 * (
						- np.log(np.linalg.det(sigma_lambda_product) + 1e-9)
						+ np.trace(sigma_lambda_product)
						+ 1.0 - p)
					KLDs[c][j] += logdet_term

					if decompose_KLD:
						KLDs_decomp[c][j,2] = logdet_term

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
		if trim_KLDs:
			logger.warning("Trimming negative KLD values, minimum of which is {:.3E}".format(np.amin(np.array(KLDs))))
			KLDs = [np.clip(kld, a_min=0.0, a_max=None) for kld in KLDs]

												 
	out = [klds / np.sum(klds) for klds in KLDs]
	rate_time = time.time() - start_time

	logger.info("The RATE calculation took {} seconds".format(round(rate_time, 3)))
	
	if decompose_KLD:
		KLDs_decomp = [pd.DataFrame(
				x,
				columns=["alpha", "mu_squared", "logdet_term"][:x.shape[1]])
				for x in KLDs_decomp]

	if C==1:
		out = out[0]
		KLDs = KLDs[0]

		if decompose_KLD:
			KLDs_decomp = KLDs_decomp[0]

	if return_KLDs:
		out = [out, KLDs]
	if return_time:
		out = [out, rate_time]
	if decompose_KLD:
		out = [out, KLDs_decomp]
	return out