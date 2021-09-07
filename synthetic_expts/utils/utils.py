from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import csv
from copy import deepcopy
from keras.callbacks import Callback as Callback
from keras import backend as K

# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
	"""Shuffle arrays in a list by shuffling each array identically."""
	num = data[0].shape[0]
	p = np.random.permutation(num)
	return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
	"""Generate batches of data.

	Given a list of array-like objects, generate batches of a given
	size by yielding a list of array-like objects corresponding to the
	same slice of each input.
	"""
	if shuffle:
		data = shuffle_aligned_list(data)

	batch_count = 0
	while True:
		if batch_count * batch_size + batch_size >= len(data[0]):
			batch_count = 0

			if shuffle:
				data = shuffle_aligned_list(data)

		start = batch_count * batch_size
		end = start + batch_size
		batch_count += 1
		yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
	"""Plot images in a grid of a given shape."""
	fig = plt.figure(1)
	grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

	size = shape[0] * shape[1]
	for i in range(size):
		grid[i].axis('off')
		grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

	plt.show()


def plot_embedding(X, y, d, title=None):
	"""Plot an embedding X with the class label y colored by the domain d."""
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)

	# Plot colors numbers
	plt.figure(figsize=(10,10))
	ax = plt.subplot(111)
	for i in range(X.shape[0]):
		# plot colored number
		plt.text(X[i, 0], X[i, 1], str(y[i]),
				 color=plt.cm.bwr(d[i] / 1.),
				 fontdict={'weight': 'bold', 'size': 9})

	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)

def categorical_hinge(y_true, y_pred):
	pos = K.sum(y_true * y_pred, axis=-1)
	neg = K.sum((1. - y_true) * y_pred, axis=-1)
	return K.maximum(0., neg - pos + 1.)

def get_paths_from_file(filepath):
	with open(filepath, 'r') as f:
		reader = csv.reader(f)
		reader.next()  # flushing first row which as directory
		image_path_with_label = []
		for row in reader:
			image_path_with_label.append(row)
		return image_path_with_label

def my_get_indexes(predicates, objects):
	"""
	input is predicates and objects
	returns the indexes of objects in predicates and -1 if it doesnt exist
	"""
	#return [predicates.index(object) for object in objects if object in predicates else -1]
	return [predicates.index(object) if object in predicates else -1 for object in objects]

def get_max_weight(predicates, iou_matrix, task_p, adv_p):
	"""
	takes input predicates, iou matrix and two tasks (one task and one adv)
	first finds indexes of these tasks in predicates
	since iou_matrix is of size predicates x predicates
	gets the sub-matrix from iou_matrix such that its task x adv
	gets the 2D coordinates of argmax of weights vector
	passes the index after shifting coordinates to iou_matrix
	returns the predicates name in x, in y, the iou weight, index in iou_matrix
	"""
	index1, index2 = my_get_indexes(predicates, task_p), my_get_indexes(predicates, adv_p)
	weights = iou_matrix[index1[0]:index1[1]+1, index2[0]:index2[1]+1]
	#print weights
	index = np.unravel_index(np.argmax(weights), weights.shape)
	#print index
	index = (index[0]+index1[0], index2[0]+index[1])
	return predicates[index[0]], predicates[index[1]], np.max(weights), index

def get_max_weight2(predicates, iou_matrix, task_p, adv_p):
	"""
	"""
	index1, index2 = my_get_indexes(predicates, task_p), my_get_indexes(predicates, adv_p)
	weights = iou_matrix[index1,:][:, index2]
	return 0,0, np.max(weights), 0

def da_get_Yadv(Ytrain, Ytest=None, guess_type='uniform'):
	# guess_type: The guess on label distribution in test data. Takes following values
	#    'uniform': all combinations of labels have same probability
	#    'uncorrelated': Closest independent(uncorrelated) distribution to train set.

	train_unique_combs = np.unique(Ytrain, axis=0)

	num_train = float(np.shape(Ytrain)[0])
	num_label = np.shape(Ytrain)[1]

	comb_dict = {}
	for c in train_unique_combs:
		comb_dict[tuple(c)] = 0.
	comb_train = deepcopy(comb_dict)
	for ytr in Ytrain:
		comb_train[tuple(ytr)] += 1./num_train

	if Ytest is None:
		comb_test = deepcopy(comb_dict)
		if guess_type == 'uniform':
			for c in comb_test:
				comb_test[c] = 1./(2**num_label) # assuming all labels are binary
		elif guess_type == 'uncorrelated':
			marginal_ytrain = np.sum(Ytrain, axis=0)/num_train
			for c in comb_test:
				marg_list = []
				for i in range(num_label):
					if c[i] == 1:
						marg_list.append(marginal_ytrain[i])
					else:
						marg_list.append(1. - marginal_ytrain[i])
				comb_test[c] = np.product(marg_list)
	else:
		num_test = float(np.shape(Ytest)[0])
		comb_test = deepcopy(comb_dict)
		for yts in Ytest:
			if tuple(yts) in comb_test:
				comb_test[tuple(yts)] += 1./num_test
	Yadv = []
	for y in Ytrain:
		ty = tuple(y)
		Yadv.append(comb_train[ty]/(comb_train[ty] + comb_test[ty]))
	return np.array(Yadv)

def da_yadv_lookup(Ytrain, Ytest=None, groups=None):
	train_unique_combs = np.unique(Ytrain, axis=0)

	num_train = float(np.shape(Ytrain)[0])
	num_label = np.shape(Ytrain)[1]

	comb_dict = {}
	for c in train_unique_combs:
		comb_dict[tuple(c)] = 0.
	comb_train = deepcopy(comb_dict)
	for ytr in Ytrain:
		comb_train[tuple(ytr)] += 1./num_train

	if Ytest is None:
		comb_group_train = {}
		for ytr in Ytrain:
			for g in groups:
				if g not in comb_group_train:
					comb_group_train[g] = {}
				if tuple(ytr[groups[g]]) in comb_group_train[g]:
					comb_group_train[g][tuple(ytr[groups[g]])] += 1./num_train
				else:
					comb_group_train[g][tuple(ytr[groups[g]])] = 0.
		Ytrain2adv = {}
		for ytr in train_unique_combs:
			p_test = 1.
			for g in groups:
				p_test *= comb_group_train[g][tuple(ytr[groups[g]])]
			Ytrain2adv[tuple(ytr)] = comb_train[tuple(ytr)]/(comb_train[tuple(ytr)] + p_test)
	else:
		num_test = float(np.shape(Ytest)[0])
		comb_test = deepcopy(comb_dict)
		for yts in Ytest:
			if tuple(yts) in comb_test:
				comb_test[tuple(yts)] += 1./num_test
		Ytrain2adv = {}
		for ytr in train_unique_combs:
			Ytrain2adv[tuple(ytr)] = comb_train[tuple(ytr)]/(comb_train[tuple(ytr)] + comb_test[tuple(ytr)])
	return Ytrain2adv

class delta_callback(Callback):
        def __init__(self,layer,lamda, predicate_groups, predicates, n_list):
                self.layer = layer
                self.predicate_groups = predicate_groups
                self.predicates = predicates
                self.n_list = n_list

        def on_train_begin(self, logs={}):
                return

        def on_train_end(self, logs={}):
                return

        def on_epoch_begin(self, epoch, logs={}):
                return

        def on_epoch_end(self, epoch, logs={}):
                weight = self.layer.get_weights()[0]
                n = compute_del_grauman(weight, self.predicate_groups, self.predicates)
                self.n_list.append(n)
                return

        def on_batch_begin(self, batch, logs={}):
                return

        def on_batch_end(self, batch, logs={}):
                return



def compute_del_grauman(weights, predicate_groups, predicates):
	# Computes the del sclaing matrix as given in eq 4 in supplementary of the CVPR 14 paper.
	delta = np.zeros(weights.shape) # initializing delta weights
	delta_sum = 0 # the normalizing factor
	for i in range(weights.shape[0]):
		for g in predicate_groups.keys():
			index = []
			for attr in predicate_groups[g]:
				index.append(predicates.index(attr))
			w_temp = weights[i, index]
			delta[i, index] = np.linalg.norm(w_temp)
			delta_sum = delta_sum + np.linalg.norm(w_temp)
	return delta/delta_sum

def L21_grauman(predicate_groups, predicates, lamda,input_dim,output_dim):
	n = np.ones((input_dim,output_dim))
	def L21(weight_matrix):
		return lamda * K.sum(K.sum(tf.div(K.square(weight_matrix),K.variable(value = n[-1]))))
	return L21
