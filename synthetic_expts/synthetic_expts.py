"""
Author: saneem
"""
import pickle
import tensorflow as tf
import numpy as np
import os, sys
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Concatenate
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import Regularizer
from keras.applications.resnet50 import ResNet50
from keras.callbacks import Callback # early stopping

import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
import time
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss

import argparse
from copy import deepcopy
from utils.utils import *
from utils.utils import da_get_Yadv
from utils.flipGrad import *
from utils.RateScheduler import GradientReversalScheduler

argparser = argparse.ArgumentParser(
	description="To train the model with our without our proposed adverserial regularizer.")

argparser.add_argument(
	'-a',
	'--algo',
	help="algorithm name: reg, base, fs, adv",
	default='base')

argparser.add_argument(
	'-hp',
	'--hyper_param',
	help="Hyper param list: will adv_weight for fs and adv and lamda for reg",
	default=[1.0],
	type=float,
	nargs='+')

argparser.add_argument(
	'-f',
	'--data_file',
	help="data directory",
	nargs='+',
	default=None)

argparser.add_argument(
	'-fs_test',
	'--tfile',
	help="what test to try",
	default=None)

args = argparser.parse_args()
num_feat = 5
num_attr = 2
comm_rep_dim = 2

# custom regularizers
lamda = 0.2
def L12_reg(weight_matrix):
	return lamda* K.sum(K.square(K.sum(K.abs(weight_matrix), axis=1)))

def Linf1_reg(weight_matrix):
	return lamda* K.sum(K.sqrt(K.max(K.abs(weight_matrix), axis=1)))

def Lh1_reg(weight_matrix):
	return lamda* K.sum(K.square(K.sum(K.sqrt(K.abs(weight_matrix)), axis=1)))

def evaluate_model(model, Xtest, Ytest):
	Ypred = model.predict(Xtest)
	if len(model.output_names) >= 2:
		Ypred = model.predict(Xtest)[model.output_names.index('task')]
	Ypred_task = Ypred[:,0]
	Ytest_task = np.reshape(Ytest[:,0], Ypred_task.shape)
	mAP =  average_precision_score(Ytest_task, Ypred_task, average='macro')
	acc = np.mean((Ypred_task > 0.5) == Ytest_task)
	return {'map':mAP, 'acc':acc, 'Ypred':Ypred, 'Ytrue':Ytest}

def save_results(model, Xtrain, Ytrain, Xtest, Ytest, filename, algo):
	res_train = evaluate_model(model, Xtrain, Ytrain)
	if type(Xtest) == dict:
		for tfile in Xtest:
			res = evaluate_model(model, Xtest[tfile], Ytest[tfile])
			for key in res_train:
				res[key+'_train'] = res_train[key]
			result_file = '../synthetic_results/' + filename.split('.pckl')[0] + '__on__' +\
							tfile.split('.pckl')[0] + '__' + algo + '.pckl'
			pickle.dump(res, open(result_file, 'w'))
	else:
		res = evaluate_model(model, Xtest, Ytest)
		for key in res_train:
			res[key+'_train'] = res_train[key]
		result_file = '../synthetic_results/' + filename.split('.pckl')[0] + '__' + algo + '.pckl'
		pickle.dump(res, open(result_file, 'w'))

def keras_basic_model(Xtrain, Ytrain, if_hdh=False):
	input_s = (np.shape(Xtrain)[1],)
	# optim = Adam(lr = 005)
	optim = SGD(lr=0.005, nesterov=True)
	num_epochs = 1000
	if len(np.shape(Ytrain)) == 1:
		num_label = 1
	else:
		num_label = np.shape(Ytrain)[1]
	# model
	earlystopper = EarlyStopping(min_delta=0.00001, patience=20, verbose=1)
	input_layer = Input(shape=input_s)
	if if_hdh:
		layer = Dense(np.shape(Xtrain)[1], activation = 'relu')(input_layer)
		output = Dense(num_label, activation='sigmoid')(layer)
	else:
		output = Dense(num_label, activation='sigmoid')(input_layer)
	model = Model(inputs=[input_layer], outputs=output)
	model.compile(loss='binary_crossentropy', #'mse',  #'categorical_crossentropy',
				optimizer=optim,
				)
	model.fit(x=Xtrain, y=Ytrain,
				epochs=num_epochs,
				batch_size=50,
				validation_data=(Xtrain, Ytrain),
				verbose=0,
				callbacks=[earlystopper])
	return model

def cross_entropy(Ytrue, Ypred):
	Ypred = np.clip(Ypred, 1e-7, 1.-1e-7)
	return np.mean(np.log(Ypred)*Ytrue + np.log(1-Ypred)*(1-Ytrue))
###################################################################################


def compete_LR(Xtrain, Ytrain):
	num_attributes = num_attr
	input_s = (num_feat*2,)
	adam = Adam(lr=0.01, decay=0.0)
	num_epochs = 2000
	data_points = len(Ytrain)

	# model
	input_layer = Input(shape=input_s)
	common_rep = Dense(comm_rep_dim, name='rep', use_bias=False)(input_layer)
	task_out = Dense(num_attributes, activation='sigmoid', name='task', kernel_regularizer=L12_reg)(common_rep)
	model = Model(inputs=[input_layer], outputs=task_out)
	model.compile(loss='binary_crossentropy', #'mse',  #'categorical_crossentropy',
				optimizer=adam,
				metrics=['binary_accuracy']
				)
	earlystopper = EarlyStopping(min_delta=0.00001, patience=20, verbose=1)
	model.fit(x=Xtrain, y=Ytrain,
				epochs=num_epochs,
				batch_size=50,
				validation_data=(Xtrain, Ytrain),
				verbose=0,
				callbacks=[earlystopper])
	# Ypred_train = model.predict(Xtrain)
	# train_acc = 1 - np.mean(abs((Ypred_train > 0.5)- Ytrain), axis=0)[0]
	#
	# Ypred_test = model.predict(Xtest)
	# test_acc = 1 - np.mean(abs((Ypred_test > 0.5)- Ytest), axis=0)[0]

	return model

def adv_wrapper_FS(Xtrain, Ytrain, adv_weight=1.0):

	def task_adv_obj(selected_feat, adv_weight=adv_weight):
		model = keras_basic_model(Xtrain[:,selected_feat], Ytrain[:,0])
		Ypred = model.predict(Xtrain[:,selected_feat])
		loss_task = cross_entropy(Ytrain[:,0], Ypred)

		model_adv = keras_basic_model(Xtrain[:,selected_feat], Ytrain[:,1])
		Ypred = model_adv.predict(Xtrain[:,selected_feat])
		loss_adv = cross_entropy(Ytrain[:,1], Ypred)

		return loss_task - adv_weight*loss_adv

	feat_num = Xtrain.shape[1]
	feat_list = range(feat_num)

	stop_flag = False
	sel_feat = []
	obj_val = np.inf
	while not stop_flag:
		min_obj = np.inf
		min_f = None
		for f in feat_list:
			feat_set = deepcopy(sel_feat)
			feat_set.append(f)
			val = task_adv_obj(feat_set)
			feat_set.remove(f)
			if val < min_obj:
				min_obj = val
				min_f = f
		if min_obj < obj_val:
			obj_val = min_obj
			sel_feat.append(min_f)
			feat_list.remove(min_f)
		else:
			stop_flag = True
		if len(feat_list) == 0:
			stop_flag = True
	return keras_basic_model(Xtrain[:,sel_feat], Ytrain), sel_feat

def logistic_regression(Xtrain, Ytrain):
	input_s = (Xtrain.shape[1],)
	optim = SGD(lr=0.01, decay=0.0)
	num_epochs = 1000
	data_points = len(Ytrain)

	# model
	input_layer_init = Input(shape=input_s)
	common_rep_init = Dense(comm_rep_dim, name='rep', use_bias=False)(input_layer_init)
	task_out_init = Dense(1, activation='sigmoid', name='task')(common_rep_init)
	model = Model(inputs=[input_layer_init], outputs=task_out_init)
	model.compile(loss='binary_crossentropy', #'mse',  #'categorical_crossentropy',
				optimizer=optim,
				metrics=['binary_accuracy']
				)
	earlystopper = EarlyStopping(min_delta=0.00001, patience=20, verbose=1)
	model.fit(x=Xtrain, y=Ytrain[:,0],
				epochs=num_epochs,
				batch_size=50,
				validation_data=(Xtrain, Ytrain[:,0]),
				verbose=1,
				callbacks=[earlystopper])
	# train_acc = model_init.evaluate(x=Xtrain, y=Ytrain[:,0], verbose=0)
	# test_acc = model_init.evaluate(x=Xtest, y=Ytest[:,0], verbose=0)
	return model

def adv_logistic(Xtrain, Ytrain, adv_weight=1.0, model_init=None):
	num_attributes = num_attr
	input_s = (num_feat*2,)
	adam = Adam(lr=0.01, decay=0.0)
	num_epochs = 2000
	data_points = len(Ytrain)

	earlystopper = EarlyStopping(min_delta=0.00001, patience=20, verbose=1)

	if model_init == None:
		# init model
		input_layer_init = Input(shape=input_s)
		common_rep_init = Dense(comm_rep_dim, name='rep', use_bias=False)(input_layer_init)
		task_out_init = Dense(1, activation='sigmoid', name='task')(common_rep_init)
		model_init = Model(inputs=[input_layer_init], outputs=task_out_init)
		model_init.compile(loss='binary_crossentropy', #'mse',  #'categorical_crossentropy',
					optimizer=adam,
					metrics=['binary_accuracy']
					)
		model_init.fit(x=Xtrain, y=Ytrain[:,0],
					epochs=num_epochs,
					batch_size=50,
					validation_data=(Xtrain, Ytrain[:,0]),
					verbose=0,
					callbacks=[earlystopper])
	# adv model.
	Flip = GradientReversal(hp_lambda=1)
	input_layer_all = Input(shape=input_s)
	common_rep_all = Dense(comm_rep_dim, name='rep', use_bias=False)(input_layer_all)
	task_out_all = Dense(1, activation='sigmoid', name='task')(common_rep_all)
	adv = Flip(common_rep_all)
	adv_out_all = Dense(1, activation='sigmoid', name='adv')(adv)
	model_all = Model(inputs=[input_layer_all], outputs=[task_out_all, adv_out_all])
	model_all.compile(loss={'task':'binary_crossentropy', 'adv':'binary_crossentropy'},
				optimizer=adam,
				metrics=['binary_accuracy'],
				loss_weights={'task':1, 'adv': adv_weight}
				)
	# copying values from init model to all model.
	model_all.layers[1].set_weights(model_init.layers[1].get_weights())
	model_all.layers[3].set_weights(model_init.layers[2].get_weights())

	# Ytrain_dict = {'task': Ytrain[:,0],
	# 			   'adv': Ytrain[:,1]}
	Ytrain_dict = {'task': Ytrain[:,0],
				   'adv': Ytrain[:,1]}
	model_all.fit(x=Xtrain, y=Ytrain_dict,
				epochs=num_epochs,
				batch_size=50,
				validation_data=(Xtrain, Ytrain_dict),
				verbose=0,
				callbacks=[earlystopper])
	return model_all

def adv_logistic_scheduler(Xtrain, Ytrain, adv_weight=1.0, increment_epochs=200):
	num_attributes = num_attr
	input_s = (num_feat*2,)
	adam = Adam(lr=0.01, decay=0.0)
	num_epochs = 1000
	data_points = len(Ytrain)

	# adv model.
	lamda_sched = K.variable(0.0, dtype='float32')
	Flip = GradientReversal(lamda_sched)

	input_layer_all = Input(shape=input_s)
	common_rep_all = Dense(comm_rep_dim, name='rep', use_bias=False)(input_layer_all)
	task_out_all = Dense(1, activation='sigmoid', name='task')(common_rep_all)
	adv = Flip(common_rep_all)
	adv_out_all = Dense(1, activation='sigmoid', name='adv')(adv)
	model_all = Model(inputs=[input_layer_all], outputs=[task_out_all, adv_out_all])
	model_all.compile(loss={'task':'binary_crossentropy', 'adv':'binary_crossentropy'},
				optimizer=adam,
				metrics=['binary_accuracy']
				)
	Ytrain_dict = {'task': Ytrain[:,0],
					'adv': Ytrain[:,1]}

	def l_scheduler(i):
		if i < int(increment_epochs):
			return  adv_weight * float(i)/increment_epochs
		else:
			return  adv_weight
	lscheduler = GradientReversalScheduler(lamda_sched, l_scheduler)

	model_all.fit(x=Xtrain, y=Ytrain_dict,
				epochs=num_epochs,
				batch_size=50,
				validation_data=(Xtrain, Ytrain_dict),
				verbose=0,
				callbacks=[lscheduler])
	return model_all

def da_logistic(Xtrain, Ytrain, adv_weight=1.0, Ytest=None, guess_type='uniform', increment_epochs=200):
	Yadv = da_get_Yadv(Ytrain, Ytest, guess_type)
	input_s = (num_feat*2,)
	adam = Adam(lr=0.01, decay=0.0)
	num_epochs = 1000
	data_points = len(Ytrain)

	# adv model.
	lamda_sched = K.variable(0.0, dtype='float32')
	Flip = GradientReversal(lamda_sched)

	input_layer_all = Input(shape=input_s)
	common_rep_all = Dense(comm_rep_dim, name='rep', use_bias=False)(input_layer_all)
	task_out_all = Dense(1, activation='sigmoid', name='task')(common_rep_all)
	adv = Flip(common_rep_all)
	adv = Dense(comm_rep_dim, activation='relu')(adv)
	adv_out_all = Dense(1, activation='sigmoid', name='adv')(adv)
	model_all = Model(inputs=[input_layer_all], outputs=[task_out_all, adv_out_all])
	model_all.compile(loss={'task':'binary_crossentropy', 'adv':'binary_crossentropy'},
				optimizer=adam,
				metrics=['binary_accuracy']
				)

	Ytrain_dict = {'task': Ytrain[:,0],
					'adv': Yadv}

	def l_scheduler(i):
		if i < int(increment_epochs):
			return  adv_weight * float(i)/increment_epochs
		else:
			return  adv_weight
	lscheduler = GradientReversalScheduler(lamda_sched, l_scheduler)

	model_all.fit(x=Xtrain, y=Ytrain_dict,
				epochs=num_epochs,
				batch_size=50,
				validation_data=(Xtrain, Ytrain_dict),
				verbose=0,
				callbacks=[lscheduler])
	return model_all

def da_balance(Xtrain, Ytrain, adv_weight=1.0, increment_epochs=200):
	Yadv = np.prod(Ytrain, axis=1) + np.prod(1-Ytrain, axis=1)
	input_s = (num_feat*2,)
	adam = Adam(lr=0.01, decay=0.0)
	num_epochs = 1000
	data_points = len(Ytrain)

	# adv model.
	lamda_sched = K.variable(0.0, dtype='float32')
	Flip = GradientReversal(lamda_sched)

	input_layer_all = Input(shape=input_s)
	common_rep_all = Dense(comm_rep_dim, name='rep', use_bias=False)(input_layer_all)
	task_out_all = Dense(1, activation='sigmoid', name='task')(common_rep_all)
	adv = Flip(common_rep_all)
	adv = Dense(5, activation='relu')(adv)
	adv = Dense(2, activation='relu')(adv)
	adv_out_all = Dense(1, activation='sigmoid', name='adv')(adv)
	model_all = Model(inputs=[input_layer_all], outputs=[task_out_all, adv_out_all])
	model_all.compile(loss={'task':'binary_crossentropy', 'adv':'binary_crossentropy'},
				optimizer=adam,
				metrics=['binary_accuracy']
				)

	Ytrain_dict = {'task': Ytrain[:,0],
					'adv': Yadv}

	def l_scheduler(i):
		if i < int(increment_epochs):
			return  adv_weight * float(i)/increment_epochs
		else:
			return  adv_weight
	lscheduler = GradientReversalScheduler(lamda_sched, l_scheduler)

	model_all.fit(x=Xtrain, y=Ytrain_dict,
				epochs=num_epochs,
				batch_size=50,
				validation_data=(Xtrain, Ytrain_dict),
				verbose=0,
				callbacks=[lscheduler])
	return model_all

def da_FS(Xtrain, Ytrain, adv_weight=1.0, Ytest=None, guess_type='uniform'):
	Yadv = da_get_Yadv(Ytrain, Ytest, guess_type)

	def da_obj(selected_feat, adv_weight=adv_weight):
		model = keras_basic_model(Xtrain[:,selected_feat], Ytrain[:,0])
		Ypred = model.predict(Xtrain[:,selected_feat])
		loss_task = cross_entropy(Ytrain[:,0], Ypred)

		model_adv = keras_basic_model(Xtrain[:,selected_feat], Yadv, if_hdh=True)
		Ypred = model_adv.predict(Xtrain[:,selected_feat])
		loss_adv = cross_entropy(Yadv, Ypred)

		return loss_task - adv_weight*loss_adv

	feat_num = Xtrain.shape[1]
	feat_list = range(feat_num)

	stop_flag = False
	sel_feat = []
	obj_val = np.inf
	while not stop_flag:
		min_obj = np.inf
		min_f = None
		for f in feat_list:
			feat_set = deepcopy(sel_feat)
			feat_set.append(f)
			val = da_obj(feat_set)
			feat_set.remove(f)
			if val < min_obj:
				min_obj = val
				min_f = f
		if min_obj < obj_val:
			obj_val = min_obj
			sel_feat.append(min_f)
			feat_list.remove(min_f)
		else:
			stop_flag = True
		if len(feat_list) == 0:
			stop_flag = True
	return keras_basic_model(Xtrain[:,sel_feat], Ytrain), sel_feat #, Xtest[:,sel_feat], Ytest)

adv_weight = 1.0

algo = args.algo
print algo
print args.hyper_param
print args.data_file

data_direc = '../iccv_synthetic_data/'
results_direc = '../iccv_synthetic_results'
train_datasets = [d for d in os.listdir(data_direc) if 'train' in d]
test_corr_expts = [d for d in os.listdir(data_direc) if 'test' in d and 'da-1.5' in d]
all_results = os.listdir(results_direc)

if args.data_file != None:
	train_datasets = [d for d in train_datasets if all(x in d for x in args.data_file)]
start_time = time.time()

for filename in train_datasets:
	print '\n'  + filename
	data = pickle.load(open(data_direc + filename))
	Xtrain = data['X']
	Ytrain = data['Y']

	# get all relevant test set
	# if filename == 'train_data_n-1000_cor-0.8_dp-1.5_da-1.5.pckl':
	if filename == 'train_data_n-1000_cor-0.8_dp-1.5_da-1.5.pckl':
		Xtest = {}
		Ytest = {}
		for tfile in test_corr_expts:
			data = pickle.load(open(data_direc + tfile))
			Xtest[tfile] = data['X']
			Ytest[tfile] = data['Y']
		print 'Test dicts loaded'
	# 		gs_filename = tfile.split('.pckl')[0] + '___gold_std.pckl'
	# 		if gs_filename not in all_results:
	# 			print 'creating gold standard for ' + tfile
	# 			model_gs = logistic_regression(Xtest[tfile], Ytest[tfile])
	# 			save_results(model_gs, Xtest[tfile], Ytest[tfile], Xtest[tfile], Ytest[tfile], tfile, '_gold_std')
	# else:
	# 	test_name = 'test_data_cor-0.5_dp' + filename.split('dp')[1]
	# 	data = pickle.load(open(data_direc + test_name))
	# 	Xtest = data['X']
	# 	Ytest = data['Y']
	# 	gs_filename = test_name.split('.pckl')[0] + '___gold_std.pckl'
	# 	if gs_filename not in all_results:
	# 		model_gs = keras_basic_model(Xtest, Ytest)
	# 		save_results(model_gs, Xtest, Ytest, Xtest, Ytest, test_name, '_gold_std')

	# base model
	if algo == 'base':
		model_base = logistic_regression(Xtrain, Ytrain)
		model_base.save('../models/' + filename.split('.pckl')[0] + '__' + algo + '.h5')
		save_results(model_base, Xtrain, Ytrain, Xtest, Ytest, filename, algo)

	# task compete regularizer
	elif algo == 'reg':
		for lamda in args.hyper_param:
			def L12_reg(weight_matrix):
				return lamda* K.sum(K.square(K.sum(K.abs(weight_matrix), axis=1)))
			model_reg = compete_LR(Xtrain, Ytrain)
			model_reg.save('../models/' + filename.split('.pckl')[0] + '__' + algo + '_lam-' + str(lamda) + '.h5')
			save_results(model_reg, Xtrain, Ytrain, Xtest, Ytest, filename, algo + '_lam-' + str(lamda))

	# adversarial FS
	elif algo == 'fs':
		for adv_weight in args.hyper_param:
			model_FS, sel_feat = adv_wrapper_FS(Xtrain, Ytrain, adv_weight)
			model_FS.save('../models/' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.h5')
			np.save('../models/Selected_features___' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.npy', sel_feat)
			save_results(model_FS, Xtrain[:,sel_feat], Ytrain, Xtest[:,sel_feat], Ytest, filename, algo + '_aw-' + str(adv_weight))

	# proposed model
	elif algo == 'adv':
		for adv_weight in args.hyper_param:
			model_adv = adv_logistic(Xtrain, Ytrain, adv_weight)
			# model_adv.save('../models/' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.h5')
			save_results(model_adv, Xtrain, Ytrain, Xtest, Ytest, filename, algo + '_aw-' + str(adv_weight))

	# proposed model with scheduler
	elif algo == 'adv_sched':
		for adv_weight in args.hyper_param:
			model_adv = adv_logistic_scheduler(Xtrain, Ytrain, adv_weight)
			# model_adv.save('../models/' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.h5')
			save_results(model_adv, Xtrain, Ytrain, Xtest, Ytest, filename, algo + '_aw-' + str(adv_weight))

	# All domain adaptation variants go here
	elif 'da' in algo:
		for adv_weight in args.hyper_param:
			if 'FS' in algo:
				if 'uni' in algo:
					model_da, sel_feat = da_FS(Xtrain, Ytrain, adv_weight, Ytest=None, guess_type='uniform')
					np.save('../models/Selected_features___' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.npy', sel_feat)
					save_results(model_da, Xtrain[:,sel_feat], Ytrain, Xtest[:,sel_feat], Ytest, filename, algo + '_aw-' + str(adv_weight))
				elif 'uncorr' in algo:
					model_da, sel_feat = da_FS(Xtrain, Ytrain, adv_weight, Ytest=None, guess_type='uncorrelated')
					np.save('../models/Selected_features___' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.npy', sel_feat)
					save_results(model_da, Xtrain[:,sel_feat], Ytrain, Xtest[:,sel_feat], Ytest, filename, algo + '_aw-' + str(adv_weight))
				elif 'exact' in algo:
					if type(Ytest) == dict:
						keys = Ytest.keys()
						if args.tfile != None:
							keys = [k for k in keys if args.tfile in k]:
						for tfile in keys:
							model_da, sel_feat = da_FS(Xtrain, Ytrain, adv_weight, Ytest=Ytest[tfile])
							np.save('../models/Selected_features___' + filename.split('.pckl')[0] + '__on__' + tfile.split('.pckl')[0] +\
									algo + '_aw-' + str(adv_weight) + '.npy', sel_feat)
							save_results(model_da, Xtrain[:,sel_feat], Ytrain, Xtest[tfile][:,sel_feat], Ytest[tfile],
									filename.split('.pckl')[0] + '__on__' + tfile, algo + '_aw-' + str(adv_weight))

					else:
						model_da, sel_feat = da_FS(Xtrain, Ytrain, adv_weight, Ytest=Ytest)
						np.save('../models/Selected_features___' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.npy', sel_feat)
						save_results(model_da, Xtrain[:,sel_feat], Ytrain, Xtest[:,sel_feat], Ytest, filename, algo + '_aw-' + str(adv_weight))
				# model_da.save('../models/' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.h5')

			else:
				if 'uni' in algo:
					model_da = da_logistic(Xtrain, Ytrain, adv_weight, Ytest=None, guess_type='uniform')
					save_results(model_da, Xtrain, Ytrain, Xtest, Ytest, filename, algo + '_aw-' + str(adv_weight))
				elif 'uncorr' in algo:
					model_da = da_logistic(Xtrain, Ytrain, adv_weight, Ytest=None, guess_type='uncorrelated')
					save_results(model_da, Xtrain, Ytrain, Xtest, Ytest, filename, algo + '_aw-' + str(adv_weight))
				elif 'exact' in algo:
					if type(Ytest) == dict:
						for tfile in Ytest:
							model_da = da_logistic(Xtrain, Ytrain, adv_weight, Ytest=Ytest[tfile])
							save_results(model_da, Xtrain, Ytrain, Xtest[tfile], Ytest[tfile],
									filename.split('.pckl')[0]+'__on__'+tfile,  algo+'_aw-'+ str(adv_weight))
					else:
						model_da = da_logistic(Xtrain, Ytrain, adv_weight, Ytest=Ytest)
						save_results(model_da, Xtrain, Ytrain, Xtest, Ytest, filename, algo + '_aw-' + str(adv_weight))
					# model_da.save('../models/' + filename.split('.pckl')[0] + '__' + algo + '_aw-' + str(adv_weight) + '.h5')

	print str(time.time() - start_time) + ' seconds'

	# screen \-d \-m python synthetic_expts.py -a adv -hp 0.2
	# screen \-d \-m python synthetic_expts.py -a adv -hp 0.4
	# screen \-d \-m python synthetic_expts.py -a adv -hp 0.6
	# screen \-d \-m python synthetic_expts.py -a adv -hp 0.8
	# screen \-d \-m python synthetic_expts.py -a adv -hp 0.9
	# screen \-d \-m python synthetic_expts.py -a adv -hp 1.0
	# screen \-d \-m python synthetic_expts.py -a adv -hp 1.1
	# screen \-d \-m python synthetic_expts.py -a adv -hp 1.5
	# screen \-d \-m python synthetic_expts.py -a adv -hp 2.0
	# screen \-d \-m python synthetic_expts.py -a adv -hp 3.0
	# screen \-d \-m python synthetic_expts.py -a adv -hp 4.0
	# screen \-d \-m python synthetic_expts.py -a fs -hp 0.5
	# screen \-d \-m python synthetic_expts.py -a fs -hp 0.8
	# screen \-d \-m python synthetic_expts.py -a fs -hp 0.85
	# screen \-d \-m python synthetic_expts.py -a fs -hp 0.9
	# screen \-d \-m python synthetic_expts.py -a fs -hp 0.95
	# screen \-d \-m python synthetic_expts.py -a fs -hp 1.0
	# screen \-d \-m python synthetic_expts.py -a fs -hp 1.05
	# screen \-d \-m python synthetic_expts.py -a fs -hp 1.1
	# screen \-d \-m python synthetic_expts.py -a fs -hp 1.2
	# screen \-d \-m python synthetic_expts.py -a fs -hp 1.5
	# screen \-d \-m python synthetic_expts.py -a fs -hp 2.0
	# screen \-d \-m python synthetic_expts.py -a reg -hp 0.2
	# screen \-d \-m python synthetic_expts.py -a reg -hp 0.5
	# screen \-d \-m python synthetic_expts.py -a reg -hp 0.7
	# screen \-d \-m python synthetic_expts.py -a reg -hp 1.0
	# screen \-d \-m python synthetic_expts.py -a reg -hp 1.2
	# screen \-d \-m python synthetic_expts.py -a reg -hp 1.4
	# screen \-d \-m python synthetic_expts.py -a reg -hp 1.8
	# screen \-d \-m python synthetic_expts.py -a reg -hp 2.0
	# screen \-d \-m python synthetic_expts.py -a reg -hp 3.0
	# screen \-d \-m python synthetic_expts.py -a reg -hp 4.0
	# screen \-d \-m python synthetic_expts.py -a reg -hp 5.0
