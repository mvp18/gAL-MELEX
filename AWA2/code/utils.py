import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.cluster.bicluster import SpectralCoclustering
import numpy as np
import csv

from loss_func import eszsl_loss_func, sje_loss_func, devise_loss_func, ale_loss_func

def rearrange_predicates(predicates, predicate_groups, groups):
    permute_predicates=[]
    for group in groups:
        permute_predicates.extend([predicates.index(att) for att in predicate_groups[group]])
    return permute_predicates

def compute_multilabel_metrics(y_true, y_pred):

    gt = []
    pred = []

    for key in y_true:
        if key is 'conc_l':
          continue
        gt.append(y_true[key])
        pred.append(y_pred[key])
    # f1 = f1_score(np.round(np.hstack(gt)), np.round(np.hstack(pred)), average='micro')
    mean_AP = average_precision_score(np.hstack(gt), np.hstack(pred), average='micro')

    return mean_AP

def compute_acc(y_pred, y_true, prior_matrix): # Not used anymore

    prior_matrix = torch.cuda.FloatTensor(prior_matrix)
    class_scores = torch.softmax(torch.mm(y_pred, prior_matrix), dim=1)
    _, predicted = torch.max(class_scores, 1)
    batch_acc = torch.sum((predicted==y_true.data)).float()

    return batch_acc

def class_averaged_top1_acc(y_true, y_pred, prior_matrix): # metric for ZSL algorithms

    class_scores = np.matmul(y_pred, prior_matrix)
    predicted_classes = np.array([np.argmax(output) for output in class_scores])
    cm = confusion_matrix(y_true, predicted_classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = sum(cm.diagonal())/prior_matrix.shape[1]

    return acc

def get_paths_from_file(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # flushing first row which is directory path
        image_path_with_label = []
        for row in reader:
            image_path_with_label.append(row[1])
        return image_path_with_label

def get_indices(predicates, objects):

    return [predicates.index(object) if object in predicates else -1 for object in objects]

def get_corr_weights(predicates, iou_matrix, task_p, adv_p):

    index1, index2 = get_indices(predicates, task_p), get_indices(predicates, adv_p)
    delta_corr = iou_matrix[index1,:][:, index2]
    max_delta_corr = np.max(delta_corr)

    # taking average delta corr of adversarial task with respect to all task in the group
    delta_corr_vector = np.average(delta_corr, axis=0)
    return max_delta_corr, delta_corr_vector

def find_class_balanced_wts(train_ids, name2class, all_classes, class2att):

    train_att_mat = np.zeros([len(train_ids), 85])
    for i in range(len(train_ids)):
        class_name = all_classes[name2class[train_ids[i]]-1]
        train_att_mat[i] = class2att[class_name]
    attr_count = {}
    for i in range(85):
        attr_count[i] = {0:len(train_ids)-np.sum(train_att_mat, axis=0)[i], 1:np.sum(train_att_mat, axis=0)[i]}

    return attr_count

def diff_corr(corr_train, corr_test):
    dis_corr = (corr_train - corr_test)
    dis_corr = np.sign(corr_train)*dis_corr
    return dis_corr.clip(0,np.inf)

def create_spectral_groups(num_clusters, random_state, predicates, all_classes, train_classes, test_classes, class2att):

    train_count_dict={}
    for i in range(len(train_classes)):
        class_name = all_classes[np.squeeze(train_classes)[i]-1]
        if class_name in train_count_dict:
            train_count_dict[class_name]+=1
        else:
            train_count_dict[class_name]=1

    test_count_dict={}
    for i in range(len(test_classes)):
        class_name = all_classes[np.squeeze(test_classes)[i]-1]
        if class_name in test_count_dict:
            test_count_dict[class_name]+=1
        else:
            test_count_dict[class_name]=1

    train_att_mat = np.zeros((85, len(train_count_dict)))
    for i,c in enumerate(train_count_dict):
        for j in range(len(class2att[c])):
            train_att_mat[j,i]=train_count_dict[c]*float(class2att[c][j])

    test_att_mat = np.array([class2att[c] for c in test_count_dict]).transpose()

    corr_train = np.corrcoef(train_att_mat)
    nans = np.isnan(corr_train)
    corr_train[nans] = 0

    corr_test = np.corrcoef(test_att_mat)
    nans = np.isnan(corr_test)
    corr_test[nans] = 0

    dis_corr = diff_corr(corr_train, corr_test)
    dis_corr += 0.0000001*np.random.rand(len(corr_train), len(corr_train))

    model = SpectralCoclustering(n_clusters=num_clusters, random_state=random_state)
    model.fit(dis_corr)

    group_dict = {}
    for i,val in enumerate(model.row_labels_):
        if 'g_' + str(val) not in group_dict:
            group_dict['g_' + str(val)] = [predicates[i]]
        else:
            group_dict['g_' + str(val)].append(predicates[i])

    return group_dict, dis_corr

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train_epoch(model, training_generator, loss_weights, optimizer, device, loss_dict, zero_shot, prior_matrix, args):

    epoch_scores = {}

    if zero_shot:
        if args.zsl_loss_func=='eszsl': loss_dict['conc_l'] = eszsl_loss_func(prior_matrix)
        if args.zsl_loss_func=='sje': loss_dict['conc_l'] = sje_loss_func(prior_matrix, args.margin)
        if args.zsl_loss_func=='devise': loss_dict['conc_l'] = devise_loss_func(prior_matrix, args.margin)
        if args.zsl_loss_func=='ale': loss_dict['conc_l'] = ale_loss_func(prior_matrix, args.margin)

    y_true = {}
    y_pred = {}

    model.train()

    runningLoss = 0

    for i, (inputs, labels) in enumerate(training_generator):

        loss = {}

        loss_all_groups = 0.0

        inputs = inputs.float().to(device)
        # Initialize gradients to zero
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Feed-forward
            output_dict = model(inputs)

            for group in loss_dict:

                if '_x_' not in group:

                    ground_truth = labels[group].detach().cpu().numpy()

                    if group=='conc_l':
                        prediction = output_dict[group].detach().cpu().numpy()
                    else:
                        prediction = torch.sigmoid(output_dict[group]).detach().cpu().numpy()

                    if group not in y_true:
                        y_true[group] = ground_truth
                    else:
                        if group=='conc_l':
                            y_true[group] = np.hstack([y_true[group], ground_truth])
                        else:
                            y_true[group] = np.vstack([y_true[group], ground_truth])

                    if group not in y_pred:
                        y_pred[group] = prediction
                    else:
                        y_pred[group] = np.vstack([y_pred[group], prediction])

                labels[group] =  labels[group].to(device)

                # Computing loss per group
                if '_x_' in group or group=='conc_l':
                    loss[group] = loss_dict[group](output_dict[group], labels[group])
                else:
                    loss[group] = loss_dict[group](torch.sigmoid(output_dict[group]), labels[group])

                if loss_weights:
                    loss_all_groups += loss[group]*loss_weights[group]

            if not loss_weights: #baseline experiment
                loss_all_groups = sum(loss.values())
            # accumulate loss
            runningLoss += loss_all_groups.item()
            # Backpropagate loss and compute gradients
            loss_all_groups.backward()
            # Update the network parameters
            optimizer.step()

    if zero_shot:
        epoch_scores['acc'] = class_averaged_top1_acc(y_true=y_true['conc_l'], y_pred=y_pred['conc_l'], prior_matrix=prior_matrix)
    else:
        epoch_scores['mAP'] = compute_multilabel_metrics(y_true=y_true, y_pred=y_pred)

    return(model, runningLoss/len(training_generator), epoch_scores)

def calculate_ec_wt(group_class_scores):

    mean_entropy = np.sum(np.array([x*np.log(x+1e-7) for x in group_class_scores]), axis=1)
    conditioning_wt = np.exp(mean_entropy)

    return conditioning_wt

def train_epoch_ec(model, training_generator, loss_weights, optimizer, device, loss_dict, adv_dict, zero_shot, prior_matrix, args):

    epoch_scores = {}

    if zero_shot:
        if args.zsl_loss_func=='eszsl': loss_dict['conc_l'] = eszsl_loss_func(prior_matrix)
        if args.zsl_loss_func=='sje': loss_dict['conc_l'] = sje_loss_func(prior_matrix, args.margin)
        if args.zsl_loss_func=='devise': loss_dict['conc_l'] = devise_loss_func(prior_matrix, args.margin)
        if args.zsl_loss_func=='ale': loss_dict['conc_l'] = ale_loss_func(prior_matrix, args.margin)

    y_true = {}
    y_pred = {}

    model.train()

    runningLoss = 0

    for i, (inputs, labels) in enumerate(training_generator):

        loss = {}

        cond_wt_dict = {}

        loss_all_groups = 0.0

        inputs = inputs.float().to(device)
        # Initialize gradients to zero
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Feed-forward
            output_dict = model(inputs)

            for group in output_dict:

                if '_x_' not in group:

                    if group=='conc_l':
                        prediction = output_dict[group].detach().cpu().numpy()
                    else:
                        prediction = torch.sigmoid(output_dict[group]).detach().cpu().numpy()

                        cond_wts_per_sample = calculate_ec_wt(prediction)

                        for adv_branch in adv_dict:
                            if adv_branch['parent'] == 'latent_'+group:
                                cond_wt_dict[adv_branch['node_name']] = torch.cuda.FloatTensor(loss_weights[adv_branch['node_name']]*cond_wts_per_sample)

                    if group not in y_pred:
                        y_pred[group] = prediction
                    else:
                        y_pred[group] = np.vstack([y_pred[group], prediction])

            for group in loss_dict:

                if '_x_' not in group:

                    ground_truth = labels[group].detach().cpu().numpy()

                    if group not in y_true:
                        y_true[group] = ground_truth
                    else:
                        if group=='conc_l':
                            y_true[group] = np.hstack([y_true[group], ground_truth])
                        else:
                            y_true[group] = np.vstack([y_true[group], ground_truth])

                labels[group] =  labels[group].to(device)
                # Computing loss per group
                if '_x_' in group or group=='conc_l':
                    loss[group] = loss_dict[group](output_dict[group], labels[group])
                else:
                    loss[group] = loss_dict[group](torch.sigmoid(output_dict[group]), labels[group])

                if '_x_' in group:
                    adv_loss = loss[group]*cond_wt_dict[group]
                    loss_all_groups += adv_loss.mean()
                else:
                    loss_all_groups += loss[group]*loss_weights[group]
            # accumulate loss
            runningLoss += loss_all_groups.item()
            # Backpropagate loss and compute gradients
            loss_all_groups.backward()
            # Update the network parameters
            optimizer.step()

    if zero_shot:
        epoch_scores['acc'] = class_averaged_top1_acc(y_true=y_true['conc_l'], y_pred=y_pred['conc_l'], prior_matrix=prior_matrix)
    else:
        epoch_scores['mAP'] = compute_multilabel_metrics(y_true=y_true, y_pred=y_pred)

    return(model, runningLoss/len(training_generator), epoch_scores)

def val_epoch(model, validation_generator, loss_dict, zero_shot, prior_matrix, args, device):

    epoch_scores = {}

    if zero_shot:
        if args.zsl_loss_func=='eszsl': loss_dict['conc_l'] = eszsl_loss_func(prior_matrix)
        if args.zsl_loss_func=='sje': loss_dict['conc_l'] = sje_loss_func(prior_matrix, args.margin)
        if args.zsl_loss_func=='devise': loss_dict['conc_l'] = devise_loss_func(prior_matrix, args.margin)
        if args.zsl_loss_func=='ale': loss_dict['conc_l'] = ale_loss_func(prior_matrix, args.margin)

    y_true = {}
    y_pred = {}

    model.eval()

    runningLoss = 0.0

    for i, (inputs, labels) in enumerate(validation_generator):

        loss = {}

        loss_all_groups = 0.0

        inputs = inputs.float().to(device)

        with torch.set_grad_enabled(False):
            # Feed-forward
            output_dict = model(inputs)

            for group in loss_dict:

                if '_x_' not in group:

                    ground_truth = labels[group].cpu().numpy()

                    if group=='conc_l':
                        prediction = output_dict[group].cpu().numpy()
                    else:
                        prediction = torch.sigmoid(output_dict[group]).cpu().numpy()

                    if group not in y_true:
                        y_true[group] = ground_truth
                    else:
                        if group=='conc_l':
                            y_true[group] = np.hstack([y_true[group], ground_truth])
                        else:
                            y_true[group] = np.vstack([y_true[group], ground_truth])

                    if group not in y_pred:
                        y_pred[group] = prediction
                    else:
                        y_pred[group] = np.vstack([y_pred[group], prediction])

                labels[group] =  labels[group].to(device)

                if '_x_' in group or group=='conc_l':
                    loss[group] = loss_dict[group](output_dict[group], labels[group]).mean()
                else:
                    loss[group] = loss_dict[group](torch.sigmoid(output_dict[group]), labels[group])

            loss_all_groups = sum(loss.values()).item()

        runningLoss += loss_all_groups

    if zero_shot==1:
        epoch_scores['acc'] = class_averaged_top1_acc(y_true=y_true['conc_l'], y_pred=y_pred['conc_l'], prior_matrix=prior_matrix)
    elif zero_shot==2 or zero_shot==3:
        epoch_scores['acc'] = class_averaged_top1_acc(y_true=y_true['conc_l'], y_pred=y_pred['conc_l'], prior_matrix=prior_matrix)
        epoch_scores['mAP'] = compute_multilabel_metrics(y_true=y_true, y_pred=y_pred)
    else:
        epoch_scores['mAP'] = compute_multilabel_metrics(y_true=y_true, y_pred=y_pred)

    return(runningLoss/len(validation_generator), epoch_scores)

def test_model(model, test_generator, device, loss_dict, zero_shot, prior_matrix):

    epoch_scores = {}

    if zero_shot:
        loss_dict['conc_l'] = eszsl_loss_func(prior_matrix)

    y_true = {}
    y_pred = {}

    model.eval()

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(test_generator):

            inputs = inputs.float().to(device)
            # Feed-forward
            output_dict = model(inputs)

            for group in loss_dict:

                if '_x_' not in group:

                    ground_truth = labels[group].cpu().numpy()

                    if group=='conc_l':
                        prediction = output_dict[group].cpu().numpy()
                    else:
                        prediction = torch.sigmoid(output_dict[group]).cpu().numpy()

                    if group not in y_true:
                        y_true[group] = ground_truth
                    else:
                        if group=='conc_l':
                            y_true[group] = np.hstack([y_true[group], ground_truth])
                        else:
                            y_true[group] = np.vstack([y_true[group], ground_truth])

                    if group not in y_pred:
                        y_pred[group] = prediction
                    else:
                        y_pred[group] = np.vstack([y_pred[group], prediction])

    if zero_shot:
        epoch_scores['acc'] = class_averaged_top1_acc(y_true=y_true['conc_l'], y_pred=y_pred['conc_l'], prior_matrix=prior_matrix)
    else:
        epoch_scores['mAP'] = compute_multilabel_metrics(y_true=y_true, y_pred=y_pred)

    return epoch_scores
