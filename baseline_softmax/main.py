import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from scipy import io
import copy
import time
import argparse
from utils import *

parser = argparse.ArgumentParser(description="Softmax Baselines on GBU Split")

parser.add_argument('-data', '--dataset', help='choose between APY/AWA2/CUB/SUN', default='AwA', type=str)
parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('-es', '--early_stop', default=20, type=int)
parser.add_argument('-lr', '--lr', default=0.001, type=float)
parser.add_argument('-bs', '--batch_size', default=128, type=int)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)

np.random.seed(0)
random_state  = np.random.RandomState(0)

args = parser.parse_args()
print('Dataset : {}\n'.format(args.dataset))

# Random seed for torch
torch.manual_seed(int(args.rand_seed))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:{}'.format(device))

res101 = io.loadmat('../xlsa17/data/'+args.dataset+'/'+'res101.mat')
att_splits=io.loadmat('../'+args.dataset+'/'+'att_splits.mat')

train_loc = 'train_loc'
val_loc = 'val_loc'
test_loc = 'test_unseen_loc'

feat = res101['features']
# Shape -> (dxN)
X_train = feat[:, np.squeeze(att_splits[train_loc]-1)].T
X_val = feat[:, np.squeeze(att_splits[val_loc]-1)].T
X_test = feat[:, np.squeeze(att_splits[test_loc]-1)].T

print('Tr:{}; Val:{}; Ts:{}\n'.format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

labels = res101['labels']
labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])

train_labels_seen = np.unique(labels_train)
val_labels_unseen = np.unique(labels_val)
test_labels_unseen = np.unique(labels_test)

i=0
for labels in train_labels_seen:
	labels_train[labels_train == labels] = i    
	i+=1

j=0
for labels in val_labels_unseen:
	labels_val[labels_val == labels] = j
	j+=1

k=0
for labels in test_labels_unseen:
	labels_test[labels_test == labels] = k
	k+=1

sig = att_splits['att']
# Shape -> (Number of attributes, Number of Classes)
train_sig = sig[:, train_labels_seen-1]
val_sig = sig[:, val_labels_unseen-1]
test_sig = sig[:, test_labels_unseen-1]

model = nn.Linear(X_train.shape[1], train_sig.shape[0])
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum=0.9)

best_val = 0.0
bestEp=1

X_train = torch.from_numpy(X_train)
X_val = torch.from_numpy(X_val)
X_test = torch.from_numpy(X_test)

labels_train = torch.from_numpy(labels_train.astype('float32'))
labels_val = torch.from_numpy(labels_val.astype('float32'))
labels_test = torch.from_numpy(labels_test.astype('float32'))

for epoch in range(args.epochs):

	if epoch-bestEp>args.early_stop:
		print("Val Score hasn't improved for {} epochs. Early Stopping by {} epochs.".format(args.early_stop, args.epochs-epoch))
		break

	since = time.time()

	rand_idx = torch.randperm(X_train.shape[0])

	X_tr = X_train[rand_idx]
	y_tr = labels_train[rand_idx]

	model, tr_loss, tr_acc = train_epoch(model, X_tr, y_tr, optimizer, train_sig, device, args)

	val_loss, val_acc = val_epoch(model, X_val, labels_val, val_sig, device, args)

	if val_acc>best_val:
		best_val = val_acc
		bestEp = epoch
		best_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since

	print('Epoch:{}/{}, Time Elapsed:{:.0f}m {:.0f}s'.format(epoch+1, args.epochs, time_elapsed // 60, time_elapsed % 60))
	print('Training loss:{}, Validation loss:{}; '.format(tr_loss, val_loss))
	print('Train Acc:{}; Validation Acc:{}'.format(tr_acc, val_acc))
	print('\n')

print('Best performance at epoch {} with Validation Acc:{}.\n'.format(bestEp+1, best_val))

print('Testing with best model')
model.load_state_dict(best_wts)
test_acc = test_model(model, X_test, labels_test, test_sig, device, args)
print('Test Acc:{}'.format(test_acc))

