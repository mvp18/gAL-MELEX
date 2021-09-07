import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn

def class_averaged_top1_acc(y_true, y_pred, prior_matrix): # metric for ZSL algorithms

    class_scores = np.matmul(y_pred, prior_matrix)
    predicted_classes = np.array([np.argmax(output) for output in class_scores])
    cm = confusion_matrix(y_true, predicted_classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = sum(cm.diagonal())/prior_matrix.shape[1]

    return acc

def softmax_loss(prior_matrix):

    def categorical_cross_entropy(y_pred, y_true):

        prior_matrix_gpu = torch.cuda.FloatTensor(prior_matrix)

        class_scores = torch.mm(y_pred, prior_matrix_gpu)

        y_true = y_true.type(torch.cuda.LongTensor) # y_true is a class index tensor of type long

        cross_entropy_loss = nn.CrossEntropyLoss()(class_scores, y_true) # Softmax inside loss function

        return cross_entropy_loss

    return categorical_cross_entropy

def train_epoch(model, X, y, optimizer, train_sig, device, args):

	model.train()
	criterion = softmax_loss(train_sig)
	running_loss = 0

	pred = np.zeros((X.shape[0], train_sig.shape[0]))

	for i in range(0, X.shape[0], args.batch_size):

		if i+args.batch_size<=X.shape[0]:
			inputs = X[i:i+args.batch_size].float().to(device)
			target = y[i:i+args.batch_size]
		else:
			inputs = X[i:].float().to(device)
			target = y[i:]

		optimizer.zero_grad()

		with torch.set_grad_enabled(True):

			output = model(inputs)

			pred[i:i+inputs.size(0)] = output.detach().cpu().numpy()

			loss = criterion(output, target)

			running_loss += loss.item()*inputs.size(0)

			loss.backward()

			optimizer.step()

	avg_loss = running_loss/X.shape[0]
	acc = class_averaged_top1_acc(y, pred, train_sig)

	return model, avg_loss, acc

def val_epoch(model, X, y, val_sig, device, args):

	model.eval()
	criterion = softmax_loss(val_sig)
	running_loss = 0

	pred = np.zeros((X.shape[0], val_sig.shape[0]))

	for i in range(0, X.shape[0], args.batch_size):

		if i+args.batch_size<=X.shape[0]:
			inputs = X[i:i+args.batch_size].float().to(device)
			target = y[i:i+args.batch_size]
		else:
			inputs = X[i:].float().to(device)
			target = y[i:]

		with torch.set_grad_enabled(False):

			output = model(inputs)

			pred[i:i+inputs.size(0)] = output.detach().cpu().numpy()

			loss = criterion(output, target)

			running_loss += loss.item()*inputs.size(0)

	avg_loss = running_loss/X.shape[0]
	acc = class_averaged_top1_acc(y, pred, val_sig)

	return avg_loss, acc

def test_model(model, X, y, test_sig, device, args):

	model.eval()

	pred = np.zeros((X.shape[0], test_sig.shape[0]))

	for i in range(0, X.shape[0], args.batch_size):

		if i+args.batch_size<=X.shape[0]:
			inputs = X[i:i+args.batch_size].float().to(device)
			target = y[i:i+args.batch_size]
		else:
			inputs = X[i:].float().to(device)
			target = y[i:]

		with torch.set_grad_enabled(False):

			output = model(inputs)

			pred[i:i+inputs.size(0)] = output.detach().cpu().numpy()

	acc = class_averaged_top1_acc(y, pred, test_sig)

	return acc
