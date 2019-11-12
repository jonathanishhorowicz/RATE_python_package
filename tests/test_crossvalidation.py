import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Reshape, Conv2D, Dense, BatchNormalization, Flatten
from tensorflow_probability.python.layers.dense_variational import DenseLocalReparameterization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Input

from rate.cross_validation import cross_validate_bnn

def network_architectures(p):
	return [
	    [
	        Dense(32, activation='relu', input_shape=(p,)),
	        DenseLocalReparameterization(1)
	    ],
	    [
	        Dense(32, activation='relu', input_shape=(p,)),
	        BatchNormalization(),
	        Dense(16, activation='relu'),
	        DenseLocalReparameterization(1)
	    ]
	]

optimizers = [lambda: Adam(1e-2), lambda: SGD(1e-4)]

def test_all():
	n, p = 100, 10
	X = np.random.randn(n, p)
	y = np.random.choice(2, size=n)
	k = 3
	n_iter = 2
	n_epochs = 3

	bnn, val_df = cross_validate_bnn(
		network_architectures(p),
		optimizers,
		"grid",
		X, y, k,
		fit_args={'epochs' : n_epochs})

	assert val_df.shape==(4, k+1)

	bnn, val_df = cross_validate_bnn(
		network_architectures(p),
		optimizers,
		"random",
		X, y, k,
		n_iter=n_iter,
		fit_args={'epochs' : n_epochs})
	
	assert val_df.shape==(n_iter, k+1)
