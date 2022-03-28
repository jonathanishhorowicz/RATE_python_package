import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import rankdata
import time
import logging
logger = logging.getLogger(__name__)

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def safe_rescale(train_arr, test_arr=None):
	# rescale training data train_arr to mean zero and variance one. Rescales
	# test_arr with the training mean/variance if provided
	#
	# Args:
	# -  train_arr: numpy array
	# - test_arr: numpy array
	#
	# Returns
	#   Rescaled arrays 
	if test_arr is not None:
		assert train_arr.shape[1]==test_arr.shape[1]
	ss = StandardScaler()
	train_arr_ = ss.fit_transform(train_arr)
	test_arr_ = ss.transform(test_arr) if test_arr is not None else None
	return train_arr_, test_arr_


def plot_learning_curves(bnn, **kwargs):
	
	# format dataframe of fit history
	df = bnn.get_fit_history()
	df["epoch"] = df.index
	df = df.melt(id_vars="epoch")
	df[["split", "metric_name"]] = df.variable.str.split("_", expand=True)

	# make plot
	g = sns.FacetGrid(df, col="metric_name", hue="split", sharey=False, **kwargs)
	g.map(sns.lineplot, "epoch", "value")
	g.add_legend()
	
	return g
	
def make_1d2d(arr):
	assert arr.ndim == 1
	return arr.reshape(arr.shape[0], 1)

def onehot_encode_labels(y):
	"""
	One-hot encode integer labels y. The number of classes is assumed to be
	the largest value in y

	Args:
		y: array with shape (n_examples,)

	Returns:
		array with shape (n_examples, n_classes)
	"""
	return OneHotEncoder(categories="auto", sparse=False).fit_transform(y.reshape(y.shape[0],1))

def get_roc_curves(variable_importances):
	"""
	Calculate ROC curves

	# TODO: set row idx as variable
	
	Args:
		variable_importances: A dataframe with the following columns:
			- method
			- n
			- p
			- repeat_idx
			- variable
	"""
	
	roc_curve_df = pd.DataFrame()
	base_fpr = np.linspace(0, 1, 101) # Interpolate tpr (y-axis) at these fpr (x-axis) values

	for method in variable_importances["method"].unique():
		for n in variable_importances["n"].unique():
			for p in variable_importances["p"].unique():
				for repeat_idx in range(np.amax(variable_importances["repeat_idx"].unique()+1)):
					df = variable_importances.loc[
						(variable_importances["method"]==method) &
						(variable_importances["repeat_idx"]==repeat_idx) &
						(variable_importances["n"]==n) &
						(variable_importances["p"]==p)
					]
					if len(df)==0:
						continue
					preds, labels = df["value"].values, df["causal"].values.astype(float)
					fpr, tpr, _ = roc_curve(labels, np.abs(preds))
					interp_tpr = np.interp(base_fpr, fpr, tpr)
					auroc = auc(fpr, tpr)
					roc_curve_df = pd.concat([
						roc_curve_df,
						pd.DataFrame({
							"fpr" : base_fpr, "tpr" : interp_tpr, "auc" : auroc,
							"method" : method, "n" : n, "p" : p
							})
					])
	return roc_curve_df

def load_mnist(fashion, onehot_encode=True, flatten_x=False, crop_x=(0,0), classes=None):
	"""
	Load the MNIST dataset

	Args:
		onehot_encode: Boolean indicating whether to one-hot encode training
						and test labels (default True)
		flatten_x: Boolean indicating whether to flatten the training and 
					test inputs to 2D arrays with shape (n_examples, image_size**2).
					If False, returned inputs have shape (n_examples, image_size, image_size
					(default False)
		crop_x: Integer controlling the size of the border to be removed from the input 
				images (default 0, meaning no cropping).
		classes: None to include all classes (default). Otherwise include a list of two 
				 integers that will be encoded as 0, 1 in the order they appear.

	Returns:
		x_train, y_train, x_test, y_test: train and test inputs and labels.
											First dimension is always the number of examples

	"""
	if not fashion:
		(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
		x_train, x_test = x_train / 255.0, x_test / 255.0
	else:
		(x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
		x_train, x_test = x_train / 255.0, x_test / 255.0  
		
	def crop(X, crop_size):
		assert crop_x[0] < X.shape[1]/2
		assert crop_x[1] < X.shape[2]/2
		
		if crop_size[0]>0:
			X = X[:,crop_size[0]:-crop_size[0],:]
		if crop_size[1]>0:
			X = X[:,:,crop_size[1]:-crop_size[1]]
		return X
	
	print(x_train.shape, x_test.shape)

	if any([x > 0 for x in crop_x]):
		x_train = crop(x_train, crop_x)
		x_test = crop(x_test, crop_x)
		
	print(x_train.shape, x_test.shape)

	# Flatten to 2d arrays (each example 1d)
	def flatten_image(X):
		return X.reshape(X.shape[0], X.shape[1]*X.shape[2])
	
	if flatten_x:
		x_train = flatten_image(x_train)
		x_test = flatten_image(x_test)

	if onehot_encode:
		y_train = onehot_encode_labels(y_train)
		y_test = onehot_encode_labels(y_test)

	if classes is not None:
		assert len(classes) == 2
		c0, c1 = classes
		train_idxs_to_keep = np.logical_or(y_train==c0, y_train==c1)
		x_train, y_train = x_train[train_idxs_to_keep,:], y_train[train_idxs_to_keep]
		test_idxs_to_keep = np.logical_or(y_test==c0, y_test==c1)
		x_test, y_test = x_test[test_idxs_to_keep,:], y_test[test_idxs_to_keep]

		y_train = (y_train==c1).astype(int)[:,np.newaxis]
		y_test = (y_test==c1).astype(int)[:,np.newaxis]

	return x_train, y_train, x_test, y_test

def make_square(arr):
	"""
	Reshape a 1D array (or 2D array with .shape[2]==1) into a square 2D array
	"""
	assert arr.ndim==1 or arr.ndim==2, "array must be 1 or 2-D"
	if arr.ndim==2:
		assert arr.shape[1]==1, "If array is 2d then second dimension must be 1"
		arr = arr.reshape(arr.shape[0])
	assert arr.shape[0]**0.5 == int(arr.shape[0]**0.5), "array shape must be square (it is {})".format(arr.shape[0])
	return arr.reshape(int(arr.shape[0]**0.5), int(arr.shape[0]**0.5))

def accuracy_onehot(labels, preds):
	"""
	Compute the accuracy of predictions using one-hot encoded labels

	Args:
		labels: array of labels with shape (n_examples, n_classes). Must be one-hot encoded
				or result may be nonsense (this is not checked)
		preds: array of predictions with shape (n_examples, n_classes)

	Returns:
		Accuracy as float. Result is in [0,1]
	"""
	assert labels.shape[0]==preds.shape[0]
	return np.sum(np.argmax(preds, axis=1) == np.argmax(labels, axis=1))/float(labels.shape[0])

def accuracy(labels, preds):
	"""
	Compute the accuracy of predictions using integer labels

	Args:
		labels: array of labels with shape (n_examples,)
		preds: array of predictions with shape (n_examples, n_classes)

	Returns:
		Accuracy as float. Result is in [0,1]
	"""
	assert labels.shape[0]==preds.shape[0]
	return np.sum(preds==labels)/float(labels.shape[0])

def get_nullify_idxs(original_size, border_size):
	"""
	Get the indices of a flattened image that lie within border_size of the 
	edge of an image (use to pass to nullify argument in RATE function)

	Args:
		original size: Integer giving the size of the image
		border_size: Integer giving the size of the border to be removed.

	Returns:
		Array of (integer) indices that lie in the border.
	"""
	assert border_size < original_size/2, "Border too large to be removed from image of this size"
	tmp = np.zeros((original_size, original_size), dtype=int)
	tmp[:border_size,:] = 1
	tmp[-border_size:,:] = 1
	tmp[:,-border_size:] = 1
	tmp[:,:border_size] = 1
	tmp = tmp.reshape(tmp.shape[0]*tmp.shape[1])
	return np.where(tmp==1)[0]

def idx2pixel(idx, image_size):
	"""
	Get the 2D pixel location corresponding to the index of its flattened array

	Args:
		idx: integer index to be converted to pixel location
		image_size: integer giving size of the image

	Returns:
		i, j: the location of the pixel corresponding to idx
	"""
	assert idx < image_size**2, "index {} too large for image size {}".format(idx, image_size)
	tmp = np.zeros(image_size**2)
	tmp[idx] = 1
	tmp = tmp.reshape(image_size, image_size)
	i, j = np.where(tmp==1)
	return i[0], j[0]

def sampled_accuracies(pred_proba_samples, labels):
	"""
	Get the sampled accuracies over the entire test set from logit samples. 

	Args:
		pred_proba_samples: array of predicted probability samples with shape
							(n_mc_samples, n_examples, n_classes)/(n_mc_samples, n_examples)
							for multiclass/binary classification. (This is the shape returned by BNN_Classifier.predict).
		labels: array of one-hot encoded labels with shape (n_examples, n_classes) for non-binary clasification
				or (n_examples,1) for binary classification.

	Returns:
		Array of test accuracies for each round of MC samples with shape (n_mc_samples,)
	"""
	binary_labels = labels.shape[1]==1
	
	assert pred_proba_samples.shape[1]==labels.shape[0], "Different number of examples in logit samples and labels"

	if not binary_labels:
		assert pred_proba_samples.shape[2]==labels.shape[1], "Different number of classes in logit samples and labels"
		sampled_test_accuracies = np.sum(
			np.argmax(pred_proba_samples, axis=2) == np.argmax(labels, axis=1)[:,np.newaxis], axis=1)/float(labels.shape[0])
		
	else:
		sampled_test_accuracies = np.sum((pred_proba_samples[:,:]>0.5) == labels[:,0], axis=1)/float(labels.shape[0])

	return sampled_test_accuracies

def accuracy_hist(pred_proba_samples, labels):
	"""
	Plot a histogram showing test accuracies.
	Just calls sampled_accuracies then plots the result.
	"""
	sampled_acc = sampled_accuracies(pred_proba_samples, labels)
	avg_accuracy = round(np.mean(sampled_acc) * 100, 3)
	print("average accuracy across " + str(pred_proba_samples.shape[0]) + " samples: " + str(avg_accuracfy) + "%\n")
	fig, ax = plt.subplots(figsize=(10,5))
	sns.distplot(100*sampled_acc, ax=ax, rug=True, kde=False)
	ax.set_xlabel("Test set accuracy (%)", fontsize=30)
	ax.set_ylabel("Frequency density", fontsize=30);
	ax.tick_params("both", labelsize=15)
	return sampled_acc

def rank_array(arr):
	assert arr.ndim==1
	return (arr.shape[0] - rankdata(arr)).astype(int)

def reverse_ranks(rankarr):
	return rankarr.shape[0] - rankarr - 1

def compute_power(pvals, SNPs):
	"""
	Compute the power for identifying causal predictors.
	Args:
		Ps: list of causal predictors
	Output: matrix with dimension (num. predictors, 2), where columns are FPR, TPR
	"""
	nsnps = len(pvals)
	all_snps = np.arange(0, nsnps)
	pos = SNPs
	negs = list(set(all_snps) - set(SNPs))

	pvals_rank = rank_array(pvals)

	rocr = np.zeros((nsnps, 2))
	for i in all_snps:
		v = pvals_rank[0:i]  # test positives
		z = list(set(all_snps) - set(v))  # test negatives

		TP = len(set(v) & set(pos))
		FP = len(set(v) & set(negs))
		TN = len(set(z) & set(negs))
		FN = len(set(z) & set(pos))

		TPR = 1.0*TP/(TP+FN); FPR = 1.0*FP/(FP+TN); #FDR = 1.0*FP/(FP+TP)

		rocr[i, :] = [FPR, TPR]

	return rocr