import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="DeViSE")

parser.add_argument('-data', '--dataset', help='choose between APY/AWA2', default='AWA2', type=str)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-mr', '--margin', default=1, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)

"""

Best Values of (lr, mr) found by validation & corr. test accuracies:

AWA2 -> (0.5, 100) -> Test Acc : 0.3305
APY  -> (2, 1)     -> Test Acc : 0.1946

"""

class DeViSE():
	
	def __init__(self, args):

		self.args = args

		self.args = args

		random.seed(self.args.rand_seed)
		np.random.seed(self.args.rand_seed)

		res101 = io.loadmat('../xlsa17/data/'+self.args.dataset+'/'+'res101.mat')
		att_splits=io.loadmat('../'+self.args.dataset+'/'+'att_splits.mat')

		train_loc = []
		val_loc = []
		test_loc = []

		if args.dataset=='APY': 
			
			cls_splits = np.load('../'+self.args.dataset+'/'+'apy_cs_split.npy', allow_pickle=True).item()
			all_classes = res101['labels']
			allclass_names=att_splits['allclasses_names']

			for i, label in enumerate(all_classes):
				if allclass_names[label-1] in cls_splits['train_cls']:
					train_loc.append(i)
				elif allclass_names[label-1] in cls_splits['val_cls']:
					val_loc.append(i)
				elif allclass_names[label-1] in cls_splits['test_cls']:
					test_loc.append(i)

		elif args.dataset=='AWA2':
			
			cls_splits = np.load('../'+self.args.dataset+'/'+'awa_cs_split.npy', allow_pickle=True).item()
			image_name_list = res101['image_files']
			image_names = np.array(['/'.join(y[0][0].split('/')[-2:]) for y in image_name_list])
			
			for i, name in enumerate(image_names):
				cls_name = name.split('/')[0]
				if cls_name in cls_splits['train_cls']:
					train_loc.append(i)
				elif cls_name in cls_splits['val_cls']:
					val_loc.append(i)
				else:
					test_loc.append(i)
		
		feat = res101['features']
		# Shape -> (dxN)
		self.X_train = feat[:, train_loc]
		self.X_val = feat[:, val_loc]
		self.X_test = feat[:, test_loc]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		self.labels_train = np.squeeze(labels[train_loc])
		self.labels_val = np.squeeze(labels[val_loc])
		self.labels_test = np.squeeze(labels[test_loc])

		train_labels_seen = np.unique(self.labels_train)
		val_labels_unseen = np.unique(self.labels_val)
		test_labels_unseen = np.unique(self.labels_test)

		i=0
		for labels in train_labels_seen:
			self.labels_train[self.labels_train == labels] = i    
			i+=1
		
		j=0
		for labels in val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j+=1
		
		k=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = k
			k+=1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

	def normalizeFeature(self, x):
	    # x = N x d (d:feature dimension, N:number of instances)
	    x = x + 1e-10
	    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	    feat = x / feature_norm[:, np.newaxis]

	    return feat

	def update_W(self, W, idx, train_classes):
		
		for j in idx:
			
			X_n = self.X_train[:, j]
			y_n = self.labels_train[j]
			y_ = train_classes[train_classes!=y_n]
			XW = np.dot(X_n, W)
			gt_class_score = np.dot(XW, self.train_sig[:, y_n])

			for i, label in enumerate(y_):
				score = self.args.margin+np.dot(XW, self.train_sig[:, label])-gt_class_score
				if score>0:
					Y = np.expand_dims(self.train_sig[:, y_n]-self.train_sig[:, label], axis=0)
					W += self.args.lr*np.dot(np.expand_dims(X_n, axis=1), Y)
					break # acc. to the authors, it was practical to stop after first margin violating term was found
		return W

	def fit(self):

		print('Training...\n')

		best_val_acc = 0.0
		best_tr_acc = 0.0
		best_val_ep = -1
		best_tr_ep = -1
		
		rand_idx = np.arange(self.X_train.shape[1])

		W = np.random.rand(self.X_train.shape[0], self.train_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		train_classes = np.unique(self.labels_train)

		for ep in range(self.args.epochs):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(W, rand_idx, train_classes)

			tr_acc = self.zsl_acc(self.X_train, W, self.labels_train, self.train_sig)			
			val_acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, val_acc, elapsed//60, elapsed%60))
			
			if val_acc>best_val_acc:
				best_val_acc = val_acc
				best_val_ep = ep+1
				best_W = np.copy(W)
			
			if tr_acc>best_tr_acc:
				best_tr_ep = ep+1
				best_tr_acc = tr_acc

			if ep+1-best_val_ep>self.args.early_stop:
				print('Early Stopping by {} epochs. Exiting...'.format(self.args.epochs-(ep+1)))
				break

		print('\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
		
		return best_W

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.array([np.argmax(output) for output in dist])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		return acc

	def evaluate(self):

		best_W = self.fit()

		print('Testing...\n')

		test_acc = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)

		print('Test Acc:{}'.format(test_acc))

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = DeViSE(args)	
	clf.evaluate()
