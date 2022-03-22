import numpy as np
from scipy.linalg import solve_triangular, sqrtm, cho_factor, cho_solve, cholesky

import logging
logger = logging.getLogger(__name__)

##########################################################
# Linear algebra
##########################################################

def chol_then_inv(A):
    c, low = cho_factor(A)
    return cho_solve((c, low), np.eye(A.shape[0]))

def qr_solve(A, b):
	"""Solve Ax=b for x using the QR decomposition"""
	Q, R = np.linalg.qr(A)
	return np.matmul(solve_triangular(R, Q.T), b)

def lstsq_solve(A, b):
	return np.linalg.lstsq(A, b, rcond=None)[0]

def chol_solve(A, b):
    c, low = cho_factor(A)
    return cho_solve((c, low), b)

# def chol_del_update(R, j):
#     # Wrapper for ``dchdex``.
#     # uses upper tringular cholesky factor

#     # Parameters
#     # ----------
#     # n : input int
#     # r : input rank-2 array('d') with bounds (ldr,*)
#     # j : input int
#     # w : input rank-1 array('d') with bounds (*)

#     # Other Parameters
#     # ----------------
#     # ldr : input int, optional
#     #     Default: shape(r,0)
#     logger.debug(f"Before Fortran call, R=\n{R}")
#     R_copy = copy.deepcopy(R) # R.copy()
#     n = R_copy.shape[0]
#     logger.debug(f"\n\nR.shape={R.shape}, n={n}, j={j}")
#     w = np.zeros(n)
#     dchdex(n, R_copy, j+1, w, n) # python -> Fortran indexing
#     return R_copy

# def chol_update_solve(chol_A, b, j):
# 	chol_A_del_j = chol_del_update(chol_A, j)
# 	return cho_solve((chol_A_del_j, False), b)

def get_solver(solver):
	if solver is None:
		return chol_solve
	elif not callable(solver):
		if isinstance(solver, str):
			if solver=="lstsq":
				return lstsq_solve
			elif solver=="qr":
				return qr_solve
			elif solver=="chol":
				return chol_solve
			# elif solver=="chol_update":
			# 	return chol_update_solve
			else:
				raise ValueError(f"Unrecognised solver {solver}")
		else:
			raise ValueError(f"Solver should be string if not callable, but is {type(solver)}")
	else:
		return solver

##########################################################
##########################################################


def kl_mvn(m0, S0, m1, S1, solver, jitter=0.0, exact=True):
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
	if exact:
		iS1S0 = solver(S1, S0)
		tr_term   = np.trace(iS1S0) # can replace this with the hadamard product
		logdet_S0 = np.linalg.slogdet(S0)
		logdet_S1 = np.linalg.slogdet(S1)

		if logdet_S0[0]!=1:
			logger.warning("logdet_S0 has sign {}".format(logdet_S0[0]))
		if logdet_S1[0]!=1:
			logger.warning("logdet_S1 has sign {}".format(logdet_S1[0]))

		logdet_term  = logdet_S1[1] - logdet_S0[1]
	else:
		tr_term = np.nan
		logdet_term = np.nan

	quad_term = diff.T @ solver(S1, diff)

	logger.debug("quad_term: {} \t tr_term: {} \t logdet_term: {}".format(
		quad_term, tr_term, logdet_term
	))

	if exact:
		kld_term = 0.5 * (tr_term + logdet_term + quad_term - N)
	else:
		kld_term = 0.5 * quad_term
		
	return quad_term, tr_term, logdet_term, kld_term

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
	
	Returns: mean and covariance of conditional distribution
	"""
	mu_cond = mu_min_j - np.dot(sigma_min_j, np.linalg.lstsq(sigma_j, mu_j, rcond=None)[0])
	#print("\n\tmu_cond: {}".format(mu_cond.shape))
	
	# is this is a low-rank update? 
	Sigma_update = np.dot(sigma_min_j, np.linalg.lstsq(sigma_j, sigma_min_j.T, rcond=None)[0])
	Sigma_cond = Sigma_min_j - Sigma_update

	
	return mu_cond, Sigma_cond