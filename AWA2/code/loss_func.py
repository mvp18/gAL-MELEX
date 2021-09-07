import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def multilabel_loss_func(weight_matrix=None, reduction=True):

    def weighted_bce(y_pred, y_true):

        if hasattr(weight_matrix, 'shape'):

            batch_size = y_pred.shape[0]

            pos_weight = weight_matrix[:batch_size, :, 0]
            neg_weight = weight_matrix[:batch_size, :, 1]

            pos_weight = torch.cuda.FloatTensor(pos_weight)
            neg_weight = torch.cuda.FloatTensor(neg_weight)

            bce = pos_weight*y_true*torch.log(y_pred + 1e-7) + neg_weight*(1 - y_true)*torch.log(1 - y_pred + 1e-7)

        else:

            bce = y_true*torch.log(y_pred + 1e-7) + (1 - y_true)*torch.log(1 - y_pred + 1e-7)

        bce = torch.sum(bce, dim=1)

        if reduction:
            loss = torch.neg(bce.mean())
        else:
            loss = torch.neg(bce)

        return loss

    return weighted_bce

def eszsl_loss_func(prior_matrix):

    def categorical_cross_entropy(y_pred, y_true):

        prior_matrix_gpu = torch.cuda.FloatTensor(prior_matrix)

        class_scores = torch.mm(y_pred, prior_matrix_gpu)

        y_true = y_true.type(torch.cuda.LongTensor) # y_true is a class index tensor of type long

        cross_entropy_loss = nn.CrossEntropyLoss()(class_scores, y_true) # Softmax inside loss function

        return cross_entropy_loss

    return categorical_cross_entropy

def sje_loss_func(prior_matrix, margin):

    def ssvm_loss(y_pred, y_true):

        prior_matrix_gpu = torch.cuda.FloatTensor(prior_matrix)

        y_true = y_true.type(torch.cuda.LongTensor) # y_true is a class index tensor of type long

        scores = torch.zeros((y_pred.shape[0], prior_matrix_gpu.shape[1]), device=torch.device("cuda:0"))
        class_scores = torch.mm(y_pred, prior_matrix_gpu)

        for i in range(scores.shape[0]):
            y_n = y_true[i].item()
            scores[i] = margin + class_scores[i] - class_scores[i, y_n]
            scores[i, y_n] = 0.0

        # loss, _ = torch.max(scores, axis=1)
        loss, _ = torch.max(scores, dim=1)

        return loss.mean()

    return ssvm_loss

def devise_loss_func(prior_matrix, margin):

    def ssvm_loss(y_pred, y_true):

        prior_matrix_gpu = torch.cuda.FloatTensor(prior_matrix)

        y_true = y_true.type(torch.cuda.LongTensor) # y_true is a class index tensor of type long

        loss = torch.zeros(y_pred.shape[0], device=torch.device("cuda:0"))
        class_scores = torch.mm(y_pred, prior_matrix_gpu)

        for i in range(loss.shape[0]):
            y_n = y_true[i].item()
            sample_score = margin + class_scores[i] - class_scores[i, y_n]
            sample_score[y_n] = 0.0
            violating_class_scores = sample_score[sample_score>0]
            if len(violating_class_scores)>0:
                loss[i] = violating_class_scores[0]

        return loss.mean()

    return ssvm_loss

def ale_loss_func(prior_matrix, margin):

    beta = np.zeros(prior_matrix.shape[1])
    for i in range(1, beta.shape[0]):
        sum_alpha=0.0
        for j in range(1, i+1):
            sum_alpha+=1/j
        beta[i] = sum_alpha

    def wsabie_loss(y_pred, y_true):

        prior_matrix_gpu = torch.cuda.FloatTensor(prior_matrix)

        y_true = y_true.type(torch.cuda.LongTensor) # y_true is a class index tensor of type long

        loss = torch.zeros(y_pred.shape[0], device=torch.device("cuda:0"))
        class_scores = torch.mm(y_pred, prior_matrix_gpu)

        for i in range(loss.shape[0]):
            y_n = y_true[i].item()
            sample_score = margin + class_scores[i] - class_scores[i, y_n]
            sample_score[y_n] = 0
            rank_y_n = len(sample_score[sample_score>0])

            if rank_y_n!=0:
                sample_wt = beta[rank_y_n]
                loss[i] = (sample_wt/rank_y_n)*sample_score.clamp(min=0).sum()

        return loss.mean()

    return wsabie_loss

def eszsl_fnl(prior_matrix):

    def frobenius_norm_loss(y_pred, y_true):

        prior_matrix_gpu = torch.cuda.FloatTensor(prior_matrix)

        y_true = y_true.type(torch.cuda.LongTensor) # y_true is a class index tensor of type long

        class_scores = torch.mm(y_pred, prior_matrix_gpu)

        Y = (-1)*torch.ones((y_true.shape[0], prior_matrix_gpu.shape[1]), device=torch.device("cuda:0"))

        for i in range(Y.shape[0]):
            Y[i, y_true[i]] = 1

        mse = nn.MSELoss(reduction='none')(class_scores, Y).sum(dim=1)
        fnl = mse**0.5

        return fnl.mean()

    return frobenius_norm_loss