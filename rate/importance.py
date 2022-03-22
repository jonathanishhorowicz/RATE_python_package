import numpy as np
import pandas as pd


from typing import Callable
from dataclasses import dataclass

import datetime
import copy

from .projections import ProjectionBase, CovarianceProjection
from .utils import isPD
from .kl_solvers import KLDSolver

import logging
logger = logging.getLogger(__name__)

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
	jitter: float = 0.0
	chunk_spec: tuple = None
	printevery: int = 0
	run_diagnostics: bool = False
	save_esa_post: bool = False
	iteration_timer: bool = False
	print_iteration_times: bool = False
	solver_maker: callable = None
  
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
	jitter: float
	chunk_spec: tuple
	printevery: int
	KLDs: np.ndarray
	grouprate : bool
	var_names: list
	diagnostic_result: pd.DataFrame
	group_sizes: list
	save_esa_post: bool
	iteration_times: list
	print_iteration_times: bool
	solver_maker: callable
		
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
	iteration_times : list

##########################################################
##########################################################

def check_posterior_shapes(m, V, X=None, xcheckdim=0):
	if m is None and V is None:
		return
	else:
		if m.ndim!=2:
			raise ValueError("mean should have 2 dimensions (has {})".format(m.ndim))
		if V.ndim!=3:
			raise ValueError("covariance should have 3 dimensions (has {})".format(V.ndim))
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
		var_names = np.array_split(var_names, n_chunks)[chunk_idx]
		if group_sizes is not None:
			group_sizes = np.array_split(group_sizes, n_chunks)[chunk_idx]
		
	if inp.Mb is None:
		Mb, Vb = inp.projection.esa_posterior(inp.X, inp.Mf, inp.Vf)
	else:
		Mb, Vb = inp.Mb, inp.Vb
		
	tracker = rateIterationTracker(
		Mb=Mb, Vb=Vb,
		projection=inp.projection,
		iteration_list=J,
		C=C, p=p,
		jitter=inp.jitter,
		chunk_spec=inp.chunk_spec,
		printevery=inp.printevery,
		KLDs=[], # np.zeros((len(J), 4)) for _ in range(C)
		grouprate=inp.groups is not None,
		var_names=var_names,
		diagnostic_result=[] if inp.run_diagnostics else None,
		group_sizes=group_sizes,
		save_esa_post=inp.save_esa_post,
		iteration_times=[] if inp.iteration_timer else None, #  np.zeros(len(J)) for _ in range(C)
		print_iteration_times=inp.print_iteration_times,
		solver_maker=inp.solver_maker
	)
	
	return tracker

def iterate_rate(tracker: rateIterationTracker):
	
	for c in range(tracker.C):

		if tracker.iteration_times is not None:
			it_start_time = datetime.datetime.now()

		# one solver object per class
		logger.debug("Creating KLDSolver instance")
		solver = tracker.solver_maker(
			tracker.Mb[c], tracker.Vb[c], tracker.iteration_list
		)

		logger.debug("Solving for KLDs")
		KLDs, diagnostics = solver.solve_all_KLDs()

		if tracker.iteration_times is not None:
			it_end_time = datetime.datetime.now()
			tracker.iteration_times.append(
				(it_end_time-it_start_time).total_seconds()
			)

		logger.debug("Storing KLD/diagnostic results")
		tracker.KLDs.append(KLDs)
		
		if tracker.diagnostic_result is not None:
			tracker.diagnostic_result.append(diagnostic_result)

		# for out_idx, j in enumerate(tracker.iteration_list):

		# 	if tracker.use_chol_update:
		# 		solve = lambda A,b,j: tracker.solver(full_chol_factor, )
		# 	else:
		# 		solver = tracker.solver

		# 	it_start_time = datetime.datetime.now()
			
		# 	if tracker.printevery!=0 and out_idx%tracker.printevery==0:
		# 		logger.info("iteration {} of {} for class {}".format(out_idx, len(tracker.iteration_list), c))

		# 	kld_out, diagnostic_out = tracker.solver()
			
			# # partition ESA posterior
			# mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j = jth_partition(tracker.Mb[c], tracker.Vb[c], j)
			
			# # conditional distribution
			# mu_cond, Sigma_cond = condition_gaussian(
			# 	mu_j, mu_min_j, sigma_j, sigma_min_j, Sigma_min_j
			# )
			
			# # calculate KLD
			# tracker.KLDs[c][out_idx] = kl_mvn(
			# 	mu_min_j, Sigma_min_j, mu_cond, Sigma_cond,
			# 	solver=solver,
			# 	jitter=tracker.jitter,
			# 	exact=tracker.exact)

			# # time per iteration and ETA estimate
			# if tracker.iteration_times is not None:
			# 	it_end_time = datetime.datetime.now()
			# 	tracker.iteration_times[c][out_idx] = (it_end_time-it_start_time).total_seconds()

			# 	if tracker.print_iteration_times:
			# 		n_remaining_it = len(tracker.iteration_list) - out_idx - 1
			# 		mean_it_time = np.mean(tracker.iteration_times[c][out_idx])
			# 		this_it_td = tracker.iteration_times[c][out_idx]
			# 		logger.info("This iteration took {}".format(str(this_it_td)))
			# 		logger.info("Remaining {} iterations will take approximately {}".format(
			# 			n_remaining_it, str(this_it_td*n_remaining_it)))

			# # get any diagnostics
			# if 
			# 	tracker.diagnostic_result[c][out_idx] = [
			# 		isPD(Sigma_min_j), np.linalg.matrix_rank(Sigma_min_j), Sigma_min_j.shape[0],
			# 		isPD(Sigma_cond), np.linalg.matrix_rank(Sigma_cond), Sigma_cond.shape[0]
			# 	]
 
			
def make_rate_result(tracker: rateIterationTracker):
	
	column_name = "group" if tracker.grouprate else "variable"
	
	# kld_dfs = [pd.DataFrame(klds, columns=["quad", "trace", "logdet", "KLD"]) for klds in tracker.KLDs]
	for df in tracker.KLDs:
		df[column_name] = tracker.var_names

		n_negative_KLDs = np.sum(df.KLD<0.0)
		if n_negative_KLDs>0:
			logger.warning(f"{n_negative_KLDs} are negative")
			df["KLD_unclipped"] = df["KLD"]

		if tracker.grouprate:
			df["group_size"] = tracker.group_sizes
			df["GroupRATE"] = df.KLD.clip(lower=0.0) * 1.0/df.group_size
			df["GroupRATE"] /= df["GroupRATE"].sum()
		else:
			df["RATE"] = df["KLD"].clip(lower=0.0)
			df["RATE"] /= df["RATE"].sum()
		
	# if tracker.diagnostic_result is not None:
	# 	tracker.diagnostic_result = [
	# 		pd.DataFrame(x, columns=["S0_pd", "S0_rank", "S0_dim", "S1_pd", "S1_rank", "S1_dim"])
	# 		for x in tracker.diagnostic_result
	# 	]

	# if this is one chunk of many we only save the ESA posterior
	# in the first chunk
	if tracker.chunk_spec is not None:
		if tracker.chunk_spec[0]>0:
			tracker.Mb = None
			tracker.Vb = None

	if not tracker.save_esa_post:
		tracker.Mb = None
		tracker.Vb = None

	# if tracker.iteration_times is not None:
	# 	tracker.iteration_times = [
	# 		pd.DataFrame({'iteration' : range(len(x)), 'time_seconds' : x}) for x in tracker.iteration_times
	# 	]
		  
	res = rateResult(
		Mb=tracker.Mb,
		Vb=tracker.Vb,
		KLDs=tracker.KLDs,
		chunk_spec=tracker.chunk_spec,
		projection=tracker.projection,
		p=tracker.p,
		C=tracker.C,
		diagnostic_result=tracker.diagnostic_result,
		iteration_times=tracker.iteration_times
	)
	
	return res

def rate(
	X=None,
	Mf=None, Vf=None,
	Mb=None, Vb=None,
	projection=None,
	groups=None,
	nullify=None,
	jitter=0.0,
	chunk_spec=None,
	printevery=0,
	run_diagnostics=False,
	save_esa_post=False,
	iteration_timer=False,
	print_iteration_times=False,
	solver_maker=None
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
		jitter=jitter,
		chunk_spec=chunk_spec,
		printevery=printevery,
		run_diagnostics=run_diagnostics,
		save_esa_post=save_esa_post,
		iteration_timer=iteration_timer,
		print_iteration_times=print_iteration_times,
		solver_maker=solver_maker
	)
	
	check_rateInput(rate_input)
	tracker = setup_rate_iterations(rate_input)
	iterate_rate(tracker)
	return make_rate_result(tracker)


