import pytest

import numpy as np

from rate.models import BnnBinaryClassifier
from rate.importance import rate, RATE_ray

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def test_ray():
	"""Test the ray result matches the sequential one
	"""

	for n, p in [[100, 10], [1000, 30], [1000, 100]]:

		X, y = make_classification(
	                n_samples=n, n_features=p, n_informative=int(0.1*p),
	                n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,
	                flip_y=0.1, shift=0.0, scale=1.0, shuffle=False, random_state=123)
		y = y[:,np.newaxis]

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

		bnn = BnnBinaryClassifier(verbose=0).fit(X_train, y_train)
		M_F, V_F = bnn.logit_posterior(X_test)

		seq_result = rate(X_test, M_F, V_F)
		ray_seq_result = RATE_ray(X_test, M_F, V_F, n_jobs=1)
		ray_par_result = RATE_ray(X_test, M_F, V_F, n_jobs=2)
		assert np.array_equal(seq_result, ray_seq_result)
		assert np.array_equal(seq_result, ray_par_result)
