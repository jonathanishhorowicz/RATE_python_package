import pytest

import numpy as np
from rate.wrapped_r import init_rate_r

def test_r_wrapping():

	n, p = 100, 10
	n_draws = 1000
	X = np.random.randn(n, p)
	f_draws = np.random.multivariate_normal(np.zeros(n), np.eye(n, n), size=(n_draws))

	rate_r = init_rate_r()
	out = rate_r(X, f_draws)
	assert out[0].shape[0]==p
	assert out[1].shape[0]==p
	assert isinstance(out[2], float)
	assert isinstance(out[3], float)