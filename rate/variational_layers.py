import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
import numpy as np

# todo: are these good initialisations?
# todo: scale factor on prior

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    model = tf.keras.Sequential([
        tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(loc=tf.zeros(n, dtype), scale_diag=tf.ones(n, dtype))
        )
    ])
    return model

def posterior_fullcov(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    model =  tf.keras.Sequential([
        tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfp.layers.MultivariateNormalTriL(n)
    ])
    return model

def posterior_mean_field(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
    ])

def densevar_layer(C, n, meanfield, beta):
    return tfp.python.layers.DenseVariational(
            C,
            use_bias=True,
            make_prior_fn=prior,
            make_posterior_fn=posterior_mean_field if meanfield else posterior_fullcov,
            kl_weight=beta/n,
            kl_use_exact=True
        )