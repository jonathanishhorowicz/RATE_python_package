import numpy as np
import pandas as pd

from scipy.linalg import solve_triangular, sqrtm

from typing import NamedTuple, Callable
from dataclasses import dataclass

from .projections import ProjectionBase, CovarianceProjection
from .utils import isPD

import logging
logger = logging.getLogger(__name__)

def qr_solve(A, b):
	"""Solve Ax=b for x using the QR decomposition
	"""
	Q, R = np.linalg.qr(A)
	return np.matmul(solve_triangular(R, Q.T), b)

@dataclass
class rateInput:
	X: np.ndarray = None
	Mf: np.ndarray = None
	Vf: np.ndarray = None
	projection: ProjectionBase = CovarianceProjection()
	Mb: np.ndarray = None
	Vb: np.ndarray = None
	nullify: list = None
	groups: list = None
	exact: bool = True
	jitter: float = 0.0
	det_fn: Callable = np.linalg.det
	chunk_spec: tuple = None
	printevery: int = 0
	run_diagnostics: bool = False
  
@dataclass
class rateIterationTracker:
	# inp: rateInput
	Mb: np.ndarray
	Vb: np.ndarray
	projection: ProjectionBase
	# this_it_idx: int
	iteration_list: list
	C: int
	p: int
	exact: bool
	jitter: float
	det_fn: Callable
	chunk_spec: tuple
	printevery: int
	KLDs: np.ndarray
	grouprate : bool
	var_names: list
	diagnostic_result: pd.DataFrame
	group_sizes: list
		
@dataclass
class rateResult:
	Mb: np.ndarray
	Vb: np.ndarray
	KLDs: pd.DataFrame
	chunk_spec: tuple
	projection: ProjectionBase
	p: int
	C: int
	diagnostic_result: pd.DataFrame
		
def check_posterior_shapes(m, V, X=None, xcheckdim=0):
	if m is None and V is None:
		return
	else:
		if m.ndim!=2:
			raise ValueError("mean should have 2 dimensions (has {})".format(m.ndim))
		if V.ndim!=3:
			raise ValueError("covariance should have 3 dimensions (has {})".format(m.ndim))
		n_classes = m.shape[0]
		dim_size = m.shape[1]
		
		if V.shape[1]!=V.shape[2]:
			raise ValueError("Covariance is not square")
		if V.shape[1]!=dim_size:
			raise ValueError("Mean and covariance shapes do not match")
			
		if X is not None:
			if X.shape[xcheckdim]!= dim_size:
				raise ValueError("X shape does not match")
		
		
def check_rateInput(inp: rateInput):
	
	# TODO: check X
	
	# check that either logit posterior or esa posterior has been specified
	if inp.Mf is None and inp.Vf is None and inp.Mb is None and inp.Vb is None:
		raise ValueError("Either specify (X,Mf,Vf) or (Mb,Vb)")
		
	# check that either logit posterior or esa posterior has been specified
	if (inp.Mf is not None and inp.Vf is not None) and (inp.Mb is not None and inp.Vb is not None):
		raise ValueError("Specify one of (X,Mf,Vf) or (Mb,Vb)")
	
	if inp.Mf is not None and inp.Vf is None or inp.Mf is None and inp.Vf is not None:
		raise ValueError("Either specify all of X,Mf,Vf or none of X,Mf,Vf")
		
	if inp.Mb is not None and inp.Vb is None or inp.Mb is None and inp.Vb is not None:
		raise ValueError("Either specify Mb and Vb or neither Mb nor Vb")
			 
	# check shapes
	check_posterior_shapes(inp.Mf, inp.Vf, inp.X, 0)
	check_posterior_shapes(inp.Mb, inp.Vb, inp.X, 1)
	
	# check groups
	if inp.groups is not None:
		unique_grouped_vars = np.unique([item for sublist in inp.groups for item in sublist])
		
		if inp.nullify is not None:
			intersection = np.intersect1d(
				unique_grouped_vars,
				inp.nullify
			)
			if intersection.shape[0]>0:
				raise ValueError("Variables in groups can't be nullified: {}".format(intersection))
			
	if inp.chunk_spec is not None:
		chunk_idx, n_chunks = inp.chunk_spec
		if chunk_idx >= n_chunks:
			raise ValueError("invalid chunk_spec {}".format(inp.chunk_spec))
			
def setup_rate_iterations(inp: rateInput):
	
	if inp.X is not None:
		p = inp.X.shape[1]
		C = inp.Mf.shape[0]
	elif inp.Mb is not None:
		C = inp.Mb.shape[0]
		p = inp.Mb.shape[1]
	else:
		raise ValueError("Could not set p and C")
		
	
	# setup the variables to be iterated over
	if inp.groups is None:
		if inp.nullify is None:
			J = [np.array([j]) for j in range(p)]
			var_names = range(p)
		else:
			J = [np.array([j]) for j in range(p) if j not in inp.nullify]
			var_names = [j for j in range(p) if j not in inp.nullify]
		group_sizes = None
	else:
		J = [np.array(g) for g in inp.groups]
		group_sizes = [len(g) for g in inp.groups]
		var_names = range(len(inp.groups))
		if inp.nullify is not None:
			J = [np.concatenate([j, inp.nullify]) for j in J]
			
	# if running as a batch job we split into chunks
	if inp.chunk_spec is not None:
		chunk_idx, n_chunks = inp.chunk_spec
		# J_chunk_idxs = np.array_split(range(len(J)), n_chunks)[chunk_idx]
		J = np.array_split(J, n_chunks)[chunk_idx]
		
	if inp.Mb is None:
		Mb, Vb = inp.projection.esa_posterior(inp.X, inp.Mf, inp.Vf)
	else:
		Mb, Vb = inp.Mb, inp.Vb
		
	tracker = rateIterationTracker(
		Mb=Mb, Vb=Vb,
		projection=inp.projection,
		#this_it_idx=0,
		iteration_list=J,
		C=C, p=p,
		exact=inp.exact,
		det_fn=inp.det_fn,
		jitter=inp.jitter,
		chunk_spec=inp.chunk_spec,
		printevery=inp.printevery,
		KLDs=[np.zeros((len(J), 4)) for _ in range(C)],
		grouprate=inp.groups is not None,
		var_names=var_names,
		diagnostic_result=[np.empty((len(J), 6), dtype=object) for _ in range(C)] if inp.run_diagnostics else None,
		group_sizes=group_sizes
	)
	
	return tracker

def iterate_rate(tracker: rateIterationTracker):
	
	for c in range(tracker.C):
		for out_idx, j in enumerate(tracker.iteration_list):
			
			if tracker.printevery!=0 and out_idx%tracker.printevery==0:
				logger.info("iteration {} of {}".format(out_idx, len(tracker.iteration_list)))
			
			# partition ESA posterior
			mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(tracker.Mb[c], tracker.Vb[c], j)
			
			# conditional distribution
			mu_cond, Sigma_cond = condition_gaussian(
				mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j
			)
			
			# calculate KLD
			tracker.KLDs[c][out_idx] = kl_mvn(
				mu_min_j, Sigma_min_j, mu_cond, Sigma_cond,
				jitter=tracker.jitter,
				det_fn=tracker.det_fn,
				exact=tracker.exact)
			
			if tracker.diagnostic_result is not None:
				tracker.diagnostic_result[c][out_idx] = [
					isPD(Sigma_min_j), np.linalg.matrix_rank(Sigma_min_j), Sigma_min_j.shape[0],
					isPD(Sigma_cond), np.linalg.matrix_rank(Sigma_cond), Sigma_cond.shape[0]
				]            
			
def make_rate_result(tracker: rateIterationTracker):
	
	column_name = "group" if tracker.grouprate else "variable"
	
	kld_dfs = [pd.DataFrame(klds, columns=["quad", "trace", "logdet", "KLD"]) for klds in tracker.KLDs]
	for df in kld_dfs:
		df[column_name] = tracker.var_names

		if tracker.grouprate:
			df["group_size"] = tracker.group_sizes
			df["GroupRATE"] = df.KLD.clip(lower=0.0) * 1.0/df.group_size
			df["GroupRATE"] /= df["GroupRATE"].sum()
		else:
			df["RATE"] = df["KLD"].clip(lower=0.0)
			df["RATE"] /= df["RATE"].sum()

		
	if tracker.diagnostic_result is not None:
		tracker.diagnostic_result = [
			pd.DataFrame(x, columns=["S0_pd", "S0_rank", "S0_dim", "S1_pd", "S1_rank", "S1_dim"])
			for x in tracker.diagnostic_result
		]
		  
	res = rateResult(
		Mb=tracker.Mb,
		Vb=tracker.Vb,
		KLDs=kld_dfs,
		chunk_spec=tracker.chunk_spec,
		projection=tracker.projection,
		p=tracker.p,
		C=tracker.C,
		diagnostic_result=tracker.diagnostic_result
	)
	
	return res

def rate(
	X=None,
	Mf=None, Vf=None,
	Mb=None, Vb=None,
	projection=None,
	groups=None,
	nullify=None,
	exact=True,
	jitter=0.0,
	det_fn=np.linalg.det,
	chunk_spec=None,
	printevery=0,
	run_diagnostics=False
):
	
	rate_input = rateInput(
		X=X,
		Mf=Mf,
		Vf=Vf,
		projection=projection,
		Mb=Mb,
		Vb=Vb,
		nullify=nullify,
		groups=groups,
		exact=exact,
		jitter=jitter,
		det_fn=det_fn,
		chunk_spec=chunk_spec,
		printevery=printevery,
		run_diagnostics=run_diagnostics
	)
	
	check_rateInput(rate_input)
	tracker = setup_rate_iterations(rate_input)
	iterate_rate(tracker)
	return make_rate_result(tracker)


def kl_mvn(m0, S0, m1, S1, jitter=0.0, det_fn=np.linalg.det, exact=True):
	"""
	Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
	Also computes KL divergence from a single Gaussian pm,pv to a set
	of Gaussians qm,qv.

	- accepts stacks of means, but only one S0 and S1
	- returns the three terms separately plus the total KLD in one list
	- also returns rank of S1 and S0 in a second list
	- can control the how the determinant is calculated using det_fn (may want to swap to pseudo-determinant)
	
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
		iS1S0 = np.linalg.lstsq(S1, S0, rcond=None)[0]
		tr_term   = np.trace(iS1S0) # can replace this with the hadamard product
		logdet_S0 = np.linalg.slogdet(S0)
		logdet_S1 = np.linalg.slogdet(S1)

		if logdet_S0[0]!=1:
			logger.warning("logdet_S0 has sign {}".format(logdet_S0[0]))
		if logdet_S1[0]!=1:
			logger.warning("logdet_S1 has sign {}".format(logdet_S1[0]))

		logdet_term  = logdet_S1[1] - logdet_S0[1]#np.log((det_fn(S1)/det_fn(S0)))# + 1e-9)
	else:
		tr_term = np.nan
		logdet_term = np.nan
	quad_term = diff.T @ np.linalg.lstsq(S1, diff, rcond=None)[0] 

	logger.debug("quad_term: {} \t tr_term: {} \t logdet_term: {}".format(
		quad_term, tr_term, logdet_term
	))
		
	return quad_term, tr_term, logdet_term, 0.5 * (tr_term + logdet_term + quad_term - N)

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