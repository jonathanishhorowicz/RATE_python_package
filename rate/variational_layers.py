import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.layers as tfpl
import numpy as np

# todo: are these good initialisations?
# todo: scale factor on the standard normal prior

def posterior_fullcov():
    
    def _fn(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        model =  tf.keras.Sequential([
            tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
            tfp.layers.MultivariateNormalTriL(n)
        ])
        return model
    return _fn

def posterior_mean_field():

    def _fn(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        model = tf.keras.Sequential([
          tfp.layers.VariableLayer(2 * n, dtype=dtype),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(loc=t[..., :n],
                         scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
              reinterpreted_batch_ndims=1)),
        ])
        return model
    return _fn

def prior_standardnormal(scale):

    def _fn(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size 
        model = tf.keras.Sequential([
            tfp.layers.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=tf.zeros(n, dtype),
                    scale_diag=scale*tf.ones(n, dtype)
                    )
            )
        ])
        return model

    return _fn

def prior_Wishart(scale):
    """Normal prior with Wishart hyperprior over non-diagoal covariance entries
    
    Recommnded by http://www.stat.columbia.edu/~gelman/research/unpublished/ms.pdf
    """
    def _fn(kernel_size, bias_size, dtype=tf.float32):
        n = kernel_size + bias_size 
        model = tf.keras.Sequential([
            tfp.layers.DistributionLambda(
                lambda t: tfd.WishartTriL(df=n+2, scale_tril=scale*tf.eye(n), input_output_cholesky=True)
            ),
            tfp.layers.DistributionLambda(
                lambda t: tfd.MultivariateNormalTriL(
                    loc=tf.zeros(n, dtype),
                    scale_tril=t
                )
            )
        ])
        return model

    return _fn

def prior_LKJ(concentration):
    """Normal prior with LKJ hyperprior over non-diagoal covariance entries
    
    May struggle for larger layers: https://discourse.mc-stan.org/t/underestimating-correlation-coefficients-with-lkj-prior/1758
    """
    def _fn(kernel_size, bias_size, dtype=tf.float32):
        n = kernel_size + bias_size 
        model = tf.keras.Sequential([
            tfp.layers.DistributionLambda(
                lambda t: tfd.CholeskyLKJ(dimension=n, concentration=concentration)
            ),
            tfp.layers.DistributionLambda(
                lambda t: tfd.MultivariateNormalTriL(
                    loc=tf.zeros(n, dtype),
                    scale_tril=t
                )
            )
        ])
        return model

    return _fn

def densevar_layer(C, n, prior_fn, post_fn, beta, kl_use_exact):
    return tfp.python.layers.DenseVariational(
            C,
            use_bias=True,
            make_prior_fn=prior_fn,
            make_posterior_fn=post_fn,
            kl_weight=beta/n,
            kl_use_exact=kl_use_exact
        )

def posterior_blockdiag(C):
    """Posterior with full-covariance posterior for each column
    of the weight matrix. The columns are independent and the biases are
    deterministic.
    
    Args:
        C: number of outputs (number of classes)
    
    """
    
    def _fn(kernel_size, bias_size, dtype=None):
        smallconst = np.log(np.expm1(1.))
        
        n_weights_block = kernel_size//C
        n_bias_block = bias_size//C

        n_weight_mean_params = n_weights_block
        n_weight_cov_params = tfp.layers.MultivariateNormalTriL.params_size(n_weights_block) - n_weights_block

        n_params_total = C*(n_weight_mean_params + n_weight_cov_params) + bias_size
        #print("{} params in total".format(n_params_total))

        block_param_indices = tf.split(np.arange(n_params_total - bias_size), C)
        split_array = [n_weight_mean_params, n_weight_cov_params]
        split_param_idxs = [tf.split(x, split_array, axis=0) for x in block_param_indices]

        model =  tf.keras.Sequential([
            tfpl.VariableLayer(n_params_total, dtype=dtype),
            tfpl.DistributionLambda(lambda t: tfd.Blockwise(
                    [
                        tfd.MultivariateNormalTriL(
                            loc=tf.gather(t,split_param_idxs[c][0], axis=-1),
                            scale_tril=tfp.math.fill_triangular(
                                1e-5 + tf.nn.softplus(smallconst + tf.gather(t,split_param_idxs[c][1], axis=-1)))
                        ) for c in range(C)
                    ] +
                    [ tfd.VectorDeterministic(loc=t[...,-bias_size:]) ]
                ) 
            )
        ])
        return model
    return _fn

# estimates variance for the biases
#
def posterior_blockdiag2(C):
    
    def _fn(kernel_size, bias_size, dtype=None):
        smallconst = np.log(np.expm1(1.))
        
        n_weights_block = kernel_size//C
        n_bias_block = bias_size//C

        n_weight_mean_params = n_weights_block
        n_weight_cov_params = tfp.layers.MultivariateNormalTriL.params_size(n_weights_block) - n_weights_block

        n_params_total = C*(n_weight_mean_params + n_weight_cov_params + 2*n_bias_block)
        #print("{} params in total".format(n_params_total))

        block_param_indices = tf.split(np.arange(n_params_total), C)
        split_array = [n_weight_mean_params, n_weight_cov_params, n_bias_block, n_bias_block]
        split_param_idxs = [tf.split(x, split_array, axis=0) for x in block_param_indices]

        model =  tf.keras.Sequential([
            tfpl.VariableLayer(n_params_total, dtype=dtype),
            tfpl.DistributionLambda(lambda t: tfd.Blockwise(
                    [
                        tfd.MultivariateNormalTriL(
                            loc=tf.gather(t,split_param_idxs[c][0], axis=-1),
                            scale_tril=tfp.math.fill_triangular(
                                1e-5 + tf.nn.softplus(smallconst + tf.gather(t,split_param_idxs[c][1], axis=-1)))
                        ) for c in range(C)
                    ] +
                    [
                        tfd.Independent(
                            tfd.Normal(loc=tf.gather(t,split_param_idxs[c][2], axis=-1),
                                       scale=1e-5 + tf.nn.softplus(smallconst + tf.gather(t,split_param_idxs[c][3], axis=-1))),
                            reinterpreted_batch_ndims=1) for c in range(C)
                    ]
                ) 
            )
        ])
        return model
    return _fn
