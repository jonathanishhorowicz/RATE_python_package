import tensorflow as tf
import numpy as np

def vanilla_gradients(model, X, numpy=True):
	"""Computes the vanilla gradients of model output w.r.t inputs.

	Args:
		model: keras model
		X: input array
		numpy: True to return numpy array, otherwise returns Tensor

	Returns:
		Gradients of the predictions w.r.t input
	"""
	X_tensor = tf.cast(X, tf.float32)

	with tf.GradientTape() as tape:
		tape.watch(X_tensor)
		preds = model(X_tensor)

	grads = tape.batch_jacobian(preds, X_tensor)
	if numpy:
		grads = grads.numpy()
	return grads

def gradient_input(model, X, numpy=True):
	"""Computes the gradients*inputs, where gradients are of model
	output wrt input

	Args:
		model: keras model
		X: input array
		numpy: True to return numpy array, otherwise returns Tensor

	Returns:
		Gradients of the predictions w.r.t input
	"""
	gradients = vanilla_gradients(model, X, False)
	gradients_inputs = tf.math.multiply(gradients, X[:,tf.newaxis,:])
	if numpy:
		gradients_inputs = gradients_inputs.numpy()
	return gradients_inputs

def integrated_gradients(model, X, n_steps=20, numpy=True):
	"""Integrated gradients using zero baseline
	
	https://keras.io/examples/vision/integrated_gradients/
	
	Args:
		model: keras model
		X: input array
		n_steps: number of interpolation steps
		numpy: True to return numpy array, otherwise returns Tensor
		
	Returns:
		Integrated gradients wrt input
	"""
	
	baseline = np.zeros(X.shape).astype(np.float32)
		
	# 1. Do interpolation.
	X = X.astype(np.float32)
	interpolated_X = [
		baseline + (step / n_steps) * (X - baseline)
		for step in range(n_steps + 1)
	]
	interpolated_X = np.array(interpolated_X).astype(np.float32)
	
	# 2. Get the gradients
	grads = []
	for i, x in enumerate(interpolated_X):
		grad = vanilla_gradients(model, x)
		grads.append(grad)
	
	# 3. Approximate the integral using the trapezoidal rule
	grads = np.array(grads)
	grads = (grads[:-1] + grads[1:]) / 2.0
	avg_grads = grads.mean(axis=0)
	
	# 4. Calculate integrated gradients and return
	integrated_grads = (X - baseline)[:,np.newaxis,:] * avg_grads
	
	return integrated_grads

def smoothed_gradients(model, X, noise=1.0, n_samples=10, numpy=True):
	"""SmoothGrad
	
	Args:
		model: keras model
		X: input array
		noise: variance of Gaussian noise added to each pixel
		n_samples: number of noisy samples
		numpy: True to return numpy array, otherwise returns Tensor
		
	Returns:
		SmoothGrad wrt input
	"""
	X = X.astype(np.float32)
	
	# 1. Add noise then get the gradients
	noisy_grads = []
	for i in range(n_samples):
		noisy_grad = vanilla_gradients(model, X + np.random.normal(0.0, noise, X.shape))
		noisy_grads.append(noisy_grad)
	noisy_grads = tf.convert_to_tensor(noisy_grads, dtype=tf.float32)
	
	# 2. Mean noisy gradient
	avg_noisy_grads = tf.reduce_mean(noisy_grads, axis=0)
	
	if numpy:
		avg_noisy_grads = avg_noisy_grads.numpy()
	return avg_noisy_grads

def guided_backprop(model, X, numpy=True):
	preds = model(X)[:,:,tf.newaxis]
	grads = vanilla_gradients(model, X, False)
	
	guided_grads = (
				tf.cast(preds > 0, "float32")
				* tf.cast(preds > 0, "float32")
				* grads
			)
	if numpy:
		guided_grads = guided_grads.numpy()
	return guided_grads