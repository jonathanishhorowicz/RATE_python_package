import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow_probability.python.layers as tfpl

import logging
logger = logging.getLogger(__name__)

# prior and posterior for final layer parameters
# these are passed to DenseVariational layer
def prior_standardnormal(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size 
    model = tfk.Sequential([
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n, dtype),
                       scale=ALPHA**0.5*tf.ones(n, dtype)),
            reinterpreted_batch_ndims=1)
        )
    ])
    return model    
    
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tfk.Sequential([
        tfp.layers.VariableLayer(2 * n,
                                 initializer=tfp.layers.BlockwiseInitializer([
                                     tf.keras.initializers.GlorotUniform(),
                                     tf.keras.initializers.Constant(POST_SCALE_VAL),
                                 ],sizes=[n, n]),
                                 dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Normal(loc=t[..., :n],
                       scale=scale_transformer(t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

    
def make_negloglik():
    # returns callable representing the negative log likelihood in the ELBO
    def negloglik(y, p_y):
        return -p_y.log_prob(y)
    return negloglik

def scale_transformer(unscaled):
    # we learn unconstrained parameters for the variances
    # this function controls how the unconstrained parameters are 
    # transformed to be strictly positive
    # return 1e-3 + tf.math.softplus(0.05 * unscaled)
    return tf.math.exp(unscaled)

def get_params(dv_layer):
    # returns means and variances from a DenseVariational Keras layer
    # reshaped appropriately - TFP stores them as a flattened vector 
    inputs = tf.ones(1) # dummy input
    param_means = tf.convert_to_tensor(value=dv_layer._posterior(inputs).mean())
    param_vars = tf.convert_to_tensor(value=dv_layer._posterior(inputs).variance())
    prev_units = dv_layer.input_spec.axes[-1]
    
    w_mean, b_mean = reshape_params(param_means, dv_layer.units, prev_units, dv_layer.use_bias)
    w_var, b_var = reshape_params(param_vars, dv_layer.units, prev_units, dv_layer.use_bias)
    
    return w_mean.numpy(), w_var.numpy(), b_mean.numpy(), b_var.numpy()

def sample_params_post(dv_layer, n_samples):
    # draw posterior samples from a DenseVariational layer
    inputs = tf.ones(1) # dummy input
    param_means = tf.convert_to_tensor(value=dv_layer._posterior(inputs).sample(n_samples))
    prev_units = dv_layer.input_spec.axes[-1]
    
    return reshape_params(param_means, dv_layer.units, prev_units, dv_layer.use_bias)

def sample_params_prior(dv_layer, n_samples):
    # draw prior samples from a DenseVariational layer
    inputs = tf.ones(1) # dummy input
    param_means = tf.convert_to_tensor(value=dv_layer._prior(inputs).sample(n_samples))
    prev_units = dv_layer.input_spec.axes[-1]
    
    return reshape_params(param_means, dv_layer.units, prev_units, dv_layer.use_bias)

def reshape_params(w, units, prev_units, use_bias):
    # go from flattened vector of parameters to weights/biases with 
    # the appropriate shapes
    if use_bias:
        split_sizes = [prev_units * units, units]
        # print(split_sizes)
        kernel, bias = tf.split(w, split_sizes, axis=-1)
        # print(kernel.shape, bias.shape)
    else:
        kernel, bias = w, None

    kernel = tf.reshape(kernel, shape=tf.concat([
        tf.shape(kernel)[:-1],
        [prev_units, units],
    ], axis=0))

    return kernel, bias

def copy_weights(bnn_model, weight_list):
    # for initialising a BNN from a trained NN with equivalent architecture
    # doesn't seem to improve the final predictions
    logger.debug("bnn model has {} layers".format(len(bnn_model.layers)))
    logger.debug("weight list has length {}".format(len(weight_list)))
    
    for l in range(len(weight_list)-1):
        bnn_model.layers[l].set_weights(weight_list[l])
        bnn_model.layers[l].trainable = False  # Freeze the layer
        
    mw, vw, mb, vb = get_params(bnn_model.layers[-2])
    logger.debug("{} {} {} {}".format(mw.shape, vw.shape, mb.shape, vb.shape))
    
    mw_init, mb_init = weight_list[-1]
    logger.debug("{} {}".format(mw_init.shape, mb_init.shape))
    
    tmp = bnn_model.layers[-2]._posterior.layers[0].get_weights()[0]
    prev_units = mw.shape[0]
    units = mw.shape[1]
    tmp[:2*prev_units:2] = mw_init[:,0]
    tmp[(prev_units*units):(prev_units*units+mb_init.shape[0])] = mb_init
    
    bnn_model.layers[-2]._posterior.layers[0].set_weights([tmp])
    
def logit_posterior(mod, X):
    # Calculate logit posterior (means and variances) given model mod and 
    # examples X
    hmodel = tfk.Model(mod.input , mod.layers[-3].output)
    Hx = hmodel(X).numpy()
    
    Mw, Vw, Mb, Vb = get_params(mod.layers[-2])
    Mw, Vw = Mw[:,[0]], Vw[:,0]
    Mb, Vb = Mb[0], Vb[0]
    Vw = np.diag(Vw)
    
    Mf = np.transpose(tf.matmul(Hx, Mw) + Mb)
    Vf = np.matmul(Hx, tf.matmul(Vw, tf.transpose(Hx))) + Vb
    
    return Mf, Vf