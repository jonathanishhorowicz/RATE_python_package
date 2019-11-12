import pytest

import numpy as np
from rate.projections import PseudoinverseProjection, CovarianceProjection
from rate.importance import rate
from rate.wrapped_r import init_rate_r

from scipy.stats import spearmanr

def psd_matrix(n):
	"""Positive semi-definite matrix with size (n,n). The matrix is full-rank"""
	V_F = np.random.randn(n, n)
	return np.dot(V_F, V_F.T)

def test_input_shape_handling():

	#
	# inconsistent ndim for M_F and V_F
	#
	with pytest.raises(ValueError):
		rate(np.random.rand(20,10), np.random.rand(20), np.random.rand(2,20,20)) # M_F and V_F n_classes don't match
	with pytest.raises(ValueError):
		rate(np.random.rand(20,10), np.random.rand(2,20), np.random.rand(20,20)) # M_F and V_F n_classes don't match

	#
	# 1D M_F, 2D V_F
	#
	with pytest.raises(ValueError):
		rate(np.random.rand(21,10), np.random.rand(20), np.random.rand(20,20)) # number of examples in X doesn't match M_F, V_F
	with pytest.raises(ValueError):
		rate(np.random.rand(20,10), np.random.rand(21), np.random.rand(20,20)) # number of examples in M_F doesn't match X, V_F
	with pytest.raises(ValueError):	
		rate(np.random.rand(20,10), np.random.rand(20), np.random.rand(21,21)) # number of examples in V_F doesn't match X, M_F
	with pytest.raises(ValueError):
		rate(np.random.rand(20,10), np.random.rand(20), np.random.rand(20,21)) # V_F not square
	rate(np.random.rand(20,10), np.random.rand(20), np.random.rand(20,20)) # should run

	#
	# 2D M_F, 3D V_F
	#
	with pytest.raises(ValueError):
		rate(np.random.rand(20,10), np.random.rand(1,20), np.random.rand(2,20,20)) # M_F and V_F n_classes don't match
	with pytest.raises(ValueError):
		rate(np.random.rand(21,10), np.random.rand(2,20), np.random.rand(2,20,20)) # number of examples in X doesn't match M_F, V_F
	with pytest.raises(ValueError):
		rate(np.random.rand(20,10), np.random.rand(2,21), np.random.rand(2,20,20)) # number of examples in M_F doesn't match X, V_F
	with pytest.raises(ValueError):	
		rate(np.random.rand(20,10), np.random.rand(2,20), np.random.rand(2,21,21)) # number of examples in V_F doesn't match X, M_F
	with pytest.raises(ValueError):
		rate(np.random.rand(20,10), np.random.rand(2,20), np.random.rand(2,20,21)) # V_F not square
	rate(np.random.rand(20,10), np.random.rand(2,20), np.random.rand(2,20,20)) # should run

def test_output_shapes_single_output_class():
	"""Check the outputs are of the right type, shape, etc when calculating RATE values for a single output class.

	Depending on argument values the returned object is different (KLDs and the computation time can be returned).
	"""
	n, p = 100, 10
	eps = 1e-9

	# Logit posterior
	M_F = np.random.rand(1,n)
	V_F = psd_matrix(n)[np.newaxis]
	X = np.random.randn(n, p)

	#
	# All the combinations that affect the returned object
	#

	# Output format: RATE (np.array)
	for proj in [CovarianceProjection(), PseudoinverseProjection()]:
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=False, return_time=False)
		assert 1.0-out.sum() <= eps
		assert isinstance(out, np.ndarray)
		assert out.shape[0]==p
		assert out.ndim==1

		# [RATE, KLDs] (both np.array)
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=True, return_time=False)
		assert len(out)==2
		assert isinstance(out[0], np.ndarray)
		assert 1.0-out[0].sum() <= eps
		assert out[0].shape[0]==p
		assert out[0].ndim==1
		assert isinstance(out[1], np.ndarray)
		assert out[1].shape[0]==p
		assert out[1].ndim==1
		assert np.array_equal(out[1]/out[1].sum(), out[0])

		# [[RATE, KLDs], time], where RATE and KLDs are np.array and time is a float
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=True, return_time=True)
		assert len(out)==2
		assert len(out[0])==2
		assert isinstance(out[0][0], np.ndarray)
		assert isinstance(out[0][1], np.ndarray)
		assert isinstance(out[1], float)
		assert 1.0-out[0][0].sum() <= eps
		assert out[0][0].shape[0]==p
		assert out[0][0].ndim==1
		assert out[0][1].shape[0]==p
		assert out[0][1].ndim==1
		assert np.array_equal(out[0][1]/out[0][1].sum(), out[0][0])

		# [RATE, time] where RATE is np.array and time is a float
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=False, return_time=True)
		assert len(out)==2
		assert isinstance(out[0], np.ndarray)
		assert out[0].shape[0]==p
		assert out[0].ndim==1
		assert 1.0-out[0].sum() <= eps
		assert isinstance(out[1], float)

def test_output_shapes_multiple_output_classes():
	"""Check the outputs are of the right type, shape, etc when calculating RATE values for a 3 output classes.

	Depending on argument values the returned object is different (KLDs and the computation time can be returned).
	"""
	n, p = 100, 10
	C = 3
	eps = 1e-9

	# Logit posterior
	M_F = np.random.randn(C,n)
	V_F = np.array([psd_matrix(n) for _ in range(C)])
	X = np.random.randn(n, p)

	#
	# All the argument combinations that affect the returned object
	#

	# Output format: RATE (list of np.array)
	for proj in [CovarianceProjection(), PseudoinverseProjection()]:
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=False, return_time=False)
		assert isinstance(out, list)
		assert len(out)==C
		assert np.all([1.0-rate_vals.sum() <= eps for rate_vals in out])
		assert np.all([isinstance(rate_vals, np.ndarray) for rate_vals in out])
		assert np.all([rate_vals.shape[0]==p for rate_vals in out])
		assert np.all([rate_vals.ndim==1 for rate_vals in out])

		# [RATE, KLDs] (both list of np.array)
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=True, return_time=False)
		assert len(out)==2

		assert isinstance(out[0], list)
		assert len(out[0])==C
		assert np.all([isinstance(rate_vals, np.ndarray) for rate_vals in out[0]])
		assert np.all([1.0-rate_vals[0].sum() for rate_vals in out[0]])
		assert np.all([rate_vals.ndim==1 for rate_vals in out[0]])
		assert np.all([rate_vals.shape[0]==p for rate_vals in out[0]])

		assert isinstance(out[1], list)
		assert len(out[1])==C
		assert np.all([isinstance(kld_vals, np.ndarray) for kld_vals in out[1]])
		assert np.all([kld_vals.ndim==1 for kld_vals in out[1]])
		assert np.all([kld_vals.shape[0]==p for kld_vals in out[1]])

		assert np.all([np.array_equal(kld_vals/kld_vals.sum(), rate_vals) for rate_vals, kld_vals in zip(out[0], out[1])])

		# [[RATE, KLDs], time], where RATE and KLDs are np.array and time is a float
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=True, return_time=True)
		assert len(out)==2
		assert len(out[0])==2
		assert isinstance(out[1], float)

		assert isinstance(out[0][0], list)
		assert len(out[0][0])==C
		assert np.all([isinstance(rate_vals, np.ndarray) for rate_vals in out[0][0]])
		assert np.all([1.0-rate_vals[0].sum() for rate_vals in out[0][0]])
		assert np.all([rate_vals.ndim==1 for rate_vals in out[0][0]])
		assert np.all([rate_vals.shape[0]==p for rate_vals in out[0][0]])

		assert isinstance(out[0][1], list)
		assert len(out[0][1])==C
		assert np.all([isinstance(kld_vals, np.ndarray) for kld_vals in out[0][1]])
		assert np.all([kld_vals.ndim==1 for kld_vals in out[0][1]])
		assert np.all([kld_vals.shape[0]==p for kld_vals in out[0][1]])

		assert np.all([np.array_equal(kld_vals/kld_vals.sum(), rate_vals) for rate_vals, kld_vals in zip(out[0][0], out[0][1])])

		# # [RATE, time] where RATE is np.array and time is a float
		out = rate(X, M_F, V_F, projection=proj, return_KLDs=False, return_time=True)
		assert len(out)==2
		assert isinstance(out[0], list)
		assert np.all([isinstance(rate_vals, np.ndarray) for rate_vals in out[0]])
		assert np.all([rate_vals.shape[0]==p for rate_vals in out[0]])
		assert np.all([rate_vals.ndim==1 for rate_vals in out[0]])
		assert np.all([1.0 - rate_vals.sum() < eps for rate_vals in out[0]])

		assert isinstance(out[1], float)

def test_rate_results():
	"""Tests that the Python code (which uses closed-forms of the effect size analogue posterior)
	converges to the same result as the original R script (which uses sampels from that posterior)
	as the number of samples increases.

	Takes about two minutes to run on Jonathan's workstation.
	"""

	n, p = 100, 10
	n_draw_vals = [1000, 3000, 10000]
	n_repeats = 10 # Need this many repeats as the variance of the RATE values from sampling can be quite large

	rate_r_func = init_rate_r()

	# Logit posterior
	M_F = np.random.rand(1,n)
	V_F = psd_matrix(n)[np.newaxis]
	X = np.random.randn(n, p)

	norms = np.zeros((len(n_draw_vals), n_repeats))

	#
	# Pseudoinverse projection
	#
	rate_python = rate(X, M_F, V_F, projection=PseudoinverseProjection()) # the python result. Doesn't use matrix factorisation

	for i, n_draws in enumerate(n_draw_vals):
		for j in range(n_repeats):

			f_draws = np.random.multivariate_normal(M_F[0], V_F[0], size=(n_draws)) # Draw samples
			rate_r, klds_r, _, _ = rate_r_func(X, f_draws, "linear", False) # Calculate rate using samples (uses R code)
			norms[i,j] = np.linalg.norm(rate_r-rate_python, ord=2) # Calculate evaluation metrics (norm, correlation)

	norm_mean = norms.mean(axis=1)
	assert np.all(norm_mean[:-1] > norm_mean[1:]) # Mean difference over repeated sets of samples should decrease

	#
	# Covariance projection
	#
	rate_python = rate(X, M_F, V_F, projection=CovarianceProjection()) # the python result. Doesn't use matrix factorisation

	for i, n_draws in enumerate(n_draw_vals):
		for j in range(n_repeats):

			f_draws = np.random.multivariate_normal(M_F[0], V_F[0], size=(n_draws)) # Draw samples
			rate_r, klds_r, _, _ = rate_r_func(X, f_draws, "covariance", False) # Calculate rate using samples (uses R code)
			norms[i,j] = np.linalg.norm(rate_r-rate_python, ord=2) # Calculate evaluation metrics (norm, correlation)

	norm_mean = norms.mean(axis=1)
	assert np.all(norm_mean[:-1] > norm_mean[1:]) # Mean difference over repeated sets of samples should decrease







	
