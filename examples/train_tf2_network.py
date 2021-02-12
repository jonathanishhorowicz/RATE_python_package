import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rate.importance import rate2
from rate.models import BnnBinaryClassifier
from rate.utils import load_mnist, make_square
from rate.projections import CovarianceProjection, PseudoinverseProjection, RidgeProjection
from rate.variational_layers import densevar_layer, prior_standardnormal, posterior_mean_field, posterior_blockdiag

import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Conv2D, Dense, BatchNormalization, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.compat.v1.keras.backend as K


import logging
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

#
# some settings
#
N_MC_SAMPLES = 5
SEED = 12345
np.random.seed(SEED)
DTYPE = np.float32

# training settings
N_EPOCHS = 2
OPTIMIZER_FN = lambda: Adam(1e-3)
BETA = 1.0
BNN_FIT_ARGS = {
    'epochs' : N_EPOCHS, # CHANGEME BACK TO 50
    'validation_split' : 0.2,
    'callbacks' : [EarlyStopping(patience=5, monitor='val_acc'), TerminateOnNaN()]
    }

#
# load data - mnist odd/even
#
X_train, y_train, X_test, y_test = load_mnist(
    False, onehot_encode=False, crop_x=5, flatten_x=True
)
y_train = y_train%2
y_test = y_test%2
X_train = X_train.astype(DTYPE)
y_train = y_train.astype(DTYPE)
X_test = X_test.astype(DTYPE)
y_test = y_test.astype(DTYPE)

#
# standardise pixels
#
ss = StandardScaler()
X_train_ = ss.fit_transform(X_train)

#
# dataset size
#
C = y_train.shape[1] if y_train.ndim==2 else 1
p = X_train.shape[1]
image_size = int(p**0.5)

def make_layers(prior, posterior):
    return [
        Reshape([image_size, image_size, 1], input_shape=(p,)),
        Conv2D(32, (5, 5), activation='relu'),
        Flatten(),
        #Dense(256, activation='relu'),
        Dense(1024, activation='relu'),
        densevar_layer(C, X_train.shape[0], prior, posterior, 1.0, False)
    ]

#
# create model
#
bnn = BnnBinaryClassifier(
    layer_fn=lambda: make_layers(
        prior_standardnormal(1.0), 
        posterior_mean_field()
    ),
    n_mc_samples=N_MC_SAMPLES,
    verbose=0)

# fit model
bnn = bnn.fit(X_train_, y_train, verbose=1, **BNN_FIT_ARGS)

# test set (standardised)
X_test_ = ss.transform(X_test)

# calculate RATE values
M_F, V_F = bnn.logit_posterior(X_test_)

rate_values_filtered = rate2(
    X_test,
    M_F, V_F,
    projection=CovarianceProjection(),
    excluded_vars=[],
    groups=[[0,1], [2,3,4,5], [6,7], list(range(8,p))],
    nullify=[])
