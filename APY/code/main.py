import torch
import torch.optim as optim
from torch.utils import data

from collections import defaultdict
import argparse
import sys
import time
import copy
import os
import numpy as np

from loss_func import multilabel_loss_func
from datagen import Dataset, load_data, load_data_custom_split
from utils import *

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main(args):

    # Seed the random states
    np.random.seed(0)
    random_state  = np.random.RandomState(0)

    # Random seed for torch
    torch.manual_seed(int(args.rand_seed))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    text_file = open('../attribute_data/attribute_names.txt','r').readlines()
    predicates = [x[:-1] for x in text_file]

    print('\nLoading data')
    img_name_list, class_list, signature_list, feature_list, attr_list, img2att, rept_img_dict = load_data()

    if args.data_split == 'gbu':
        print('\nLoading data with GBU splits')
        img_name_list, class_list, signature_list, feature_list, attr_list, img2att, rept_img_dict = load_data()
        ds_type = '' # This is used later while createing folder names
    elif args.data_split == 'cs':
        new_split_file = '../apy_cs_split.npy'
        print('\Loading data with new splits as in ' + new_split_file)
        img_name_list, class_list, signature_list, feature_list, attr_list, img2att, rept_img_dict = load_data_custom_split(new_split_file)
        ds_type = '_corr_shift'


    train_ids = img_name_list[0]
    val_ids = img_name_list[1]
    test_ids = img_name_list[2]

    print('\n#Train images = {}'.format(len(train_ids)))
    print('#Val images = {}'.format(len(val_ids)))
    print('#Test images = {}'.format(len(test_ids)))

    ''' For zero shot learning'''

    train_classes = np.unique(class_list[0]).tolist()
    val_classes = np.unique(class_list[1]).tolist()
    test_classes = np.unique(class_list[2]).tolist()

    print('\n#Train Classes = {}'.format(len(train_classes)))
    print('#Val Classes = {}'.format(len(val_classes)))
    print('#Test Classes = {}'.format(len(test_classes)))

    spectral_groups, attr_distance = create_spectral_groups(num_clusters=args.clusters, random_state=random_state, train_img_names=img_name_list[0],
                                                            img2att=img2att, rept_img_dict=rept_img_dict, test_prior=signature_list[2],
                                                            predicates=predicates)

    #spectral groups
    predicate_groups = spectral_groups
    print('\nUsing {} attribute groups formed by spectral co-clustering.'.format(len(predicate_groups)))

    if args.groups == ['all']:
        groups = predicate_groups.keys()
    elif args.groups == ['']:
        groups = [] # no group heads created
    else:
        groups = args.groups

    if args.adv_list == ['all']:
        adv_list = predicate_groups.keys()
    else:
        adv_list = args.adv_list

    permute_predicates = rearrange_predicates(predicates, predicate_groups, groups)
    # Rearrange the attribute indices of prior matrix according to the new grouping
    prior_matrix_tr = signature_list[0][permute_predicates, :]
    prior_matrix_v = signature_list[1][permute_predicates, :]
    prior_matrix_ts = signature_list[2][permute_predicates, :]

    ''' '''

    if args.latent_size_after == [] and args.adv_list != []:
        print('\nWARNING: Group heads have nothing to latch on to, add -lsa vector')
        sys.exit()

    vector_size = [list(map(int, args.latent_size_before)), list(map(int, args.latent_size_after))]
    print('\nVector size:{}'.format(vector_size))

    loss_dict ={}

    if args.balance:
        print('\nUsing weighted bce loss')
        weight_variable = {}
        bal_weights =  find_class_balanced_wts(train_img_names=img_name_list[0], attr_labels_dict=attr_list[0])

        for group in groups:
            w_dict = np.array([[bal_weights[predicates.index(att)][1] for att in predicate_groups[group]],
            [bal_weights[predicates.index(att)][0] for att in predicate_groups[group]]]).transpose().astype('float')
            w_dict = w_dict/w_dict.sum(axis=1)[:, np.newaxis]
            w_dict[:,[0,1]] = w_dict[:,[1,0]]
            w_dict = np.repeat(np.expand_dims(w_dict, 0), args.batch_size, axis=0)
            weight_variable[group] = w_dict

            if args.zero_shot!=1:
                loss_dict[group] = multilabel_loss_func(weight_matrix=w_dict, reduction=True)
    else:
        if args.zero_shot!=1:
            print('\nLoss used is simple binary cross_entropy')
            for group in groups:
                loss_dict[group] = multilabel_loss_func(weight_matrix=None, reduction=True)

    if not args.baseline:
        print('\nModel will have adversarial branches for decorrelation')
        adv_candidates = []
        for gr in groups:
            for adv in adv_list:
                if adv != gr:
                    adict = {'node_name': gr + '_x_' + adv,
                            'parent': 'latent_' + gr,
                            'group': adv
                            }
                    adv_candidates.append(adict)

        if args.adv_balance==1: #Not doing this experiment anymore
            print('\nUsing Jaccard similarity weights for adversarial branches\n')
            iou_matrix = np.load('../CUB_training_attr_sim_iou.npy')
        if args.adv_balance==2:
            print('\nUsing delta(corr) weights for adversarial branches\n')
            iou_matrix = attr_distance
        if args.adv_balance:
            iou_matrix[np.diag_indices_from(iou_matrix)] = 0.0
            iou_matrix[np.isnan(iou_matrix)] = 0.0
        else:
            print('\nNo weights used for any adversarial branch\n')

        loss_weights_unnorm = []
        loss_weights = {}
        adv_dict = []

        if args.adv_balance:
            for adv_branch in adv_candidates:
                max_delta_corr, delta_corr_vector = get_corr_weights(predicates, iou_matrix, predicate_groups[adv_branch['parent'][7:]], predicate_groups[adv_branch['group']])
                delta_corr_weights = np.repeat(np.expand_dims(delta_corr_vector, 1), 2, axis=1)
                delta_corr_weights = np.repeat(np.expand_dims(delta_corr_weights, 0), args.batch_size, axis=0)

                if max_delta_corr >= args.adv_threshold:
                    loss_weights_unnorm.append((adv_branch['node_name'], max_delta_corr))
                    adv_dict.append(adv_branch)

                    if args.balance:
                        weight_matrix = weight_variable[adv_branch['group']]
                    else:
                        weight_matrix = np.ones((args.batch_size, len(predicate_groups[adv_branch['group']]), 2))

                    if args.individual_adv_weight:
                        weight_matrix = weight_matrix * delta_corr_weights

                    loss_dict[adv_branch['node_name']] = multilabel_loss_func(weight_matrix=weight_matrix, reduction=bool(1-args.entropy_cond))
        else:
            adv_dict = adv_candidates
            for adv_branch in adv_dict:
                loss_weights_unnorm.append((adv_branch['node_name'], 1.0))
                if args.balance:
                    loss_dict[adv_branch['node_name']] = multilabel_loss_func(weight_matrix=weight_variable[adv_branch['group']],
                                                                              reduction=bool(1-args.entropy_cond))
                else:
                    loss_dict[adv_branch['node_name']] = multilabel_loss_func(weight_matrix=None, reduction=bool(1-args.entropy_cond))


        print('Total #adversarial-arms={}\n'.format(len(adv_dict)))
        su = float(sum([x[1] for x in loss_weights_unnorm]))
        loss_weights.update(dict([(x[0], x[1]/su) for x in loss_weights_unnorm]))
    else:
        print('\nBaseline experiment - no adversarial tasks')
        loss_weights = None
        adv_dict = []

    if loss_weights:
        if args.zero_shot:
            print('Zero shot learning with adversarial tasks\n')
            if args.adv_balance:
                loss_weights['conc_l'] = args.main_task_weight
            else:
                loss_weights['conc_l'] = 1.0
            if args.zero_shot==2: # For a later time
                print('Primary Group Attribute Prediction Loss + Zero-Shot Categorical Loss\n')
        if args.zero_shot!=1:
            for gr in groups:
                if args.adv_balance:
                    loss_weights[gr] = args.main_task_weight
                else:
                    loss_weights[gr] = 1.0
        print('Loss weights : {}\n'.format(loss_weights))

    train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
    eval_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 6}

    training_set = Dataset(data_dict=feature_list[0], labels_dict=attr_list[0], class_list=class_list[0], eszsl_classes=train_classes, list_IDs=train_ids,
                           attributes=predicates, attribute_groups=predicate_groups, groups=groups, adv_dict=adv_dict, zero_shot=args.zero_shot)
    training_generator = data.DataLoader(training_set, **train_params)

    validation_set = Dataset(data_dict=feature_list[1], labels_dict=attr_list[1], class_list=class_list[1], eszsl_classes=val_classes, list_IDs=val_ids,
                             attributes=predicates, attribute_groups=predicate_groups, groups=groups, adv_dict=adv_dict, zero_shot=args.zero_shot)
    validation_generator = data.DataLoader(validation_set, **eval_params)

    test_set = Dataset(data_dict=feature_list[2], labels_dict=attr_list[2], class_list=class_list[2], eszsl_classes=test_classes, list_IDs=test_ids,
                       attributes=predicates, attribute_groups=predicate_groups, groups=groups, adv_dict=adv_dict, zero_shot=args.zero_shot)
    test_generator = data.DataLoader(test_set, **eval_params)

    if args.baseline or args.grad_reversal_method=='neg-grad':
        from model import GAL
    else:
        print('Invalid model config specified')
        sys.exit()

    model = GAL(predicate_groups=predicate_groups, groups=groups, vector_size=vector_size, proj_version=args.proj_version, ec=args.entropy_cond,
                adv_dict=adv_dict, loss_weights=loss_weights, zero_shot=args.zero_shot, drop_rate=args.drop_rate, add_relu=args.add_relu)
    model.apply(init_weights)
    model.to(device)

    ''' Pick which layers to optimize and specify special configs for each (if any)'''

    params_to_optimize=[]
    if len(vector_size[0])>0:
        params_to_optimize.append({'params':model.lsb.parameters()})

    params_to_optimize.append({'params':model.group_heads.parameters(), 'lr':args.learning_rate})

    if len(vector_size[1])>1:
        params_to_optimize.append({'params':model.lsa_main_dict.parameters(), 'lr':args.learning_rate})
        if adv_dict:
            params_to_optimize.append({'params':model.lsa_adv_dict.parameters()})

    params_to_optimize.append({'params':model.group_class_scores.parameters(), 'weight_decay':args.dense_reg, 'lr':args.learning_rate})

    if adv_dict:
        params_to_optimize.append({'params':model.adv_class_scores.parameters(), 'weight_decay':args.dense_reg})

    ''' '''

    if args.optimizer:
        optimizer = optim.Adam(params_to_optimize, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.SGD(params_to_optimize, lr=args.learning_rate, nesterov=True, momentum=0.9)

    if args.baseline:
        print('\nLearning rate scheduler being used for a baseline experiment\n')
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True, threshold=1e-3, min_lr=1e-5)

    def lambda_scheduler(i):
        if i < int(args.increment_epochs):
            return  args.adv_task_weight *(2. / (1. + np.exp(-3. * (float(i) / args.increment_epochs))) - 1)
        else:
            return  args.adv_task_weight

    best_val = 0.0
    training_loss=[]
    valid_loss=[]

    if not args.zero_shot:
        train_ap=[]
        val_ap=[]
        test_ap = []
    else:
        train_acc=[]
        val_acc=[]
        test_acc = []

    if args.entropy_cond and not args.baseline:
        print('Using entropy conditioning to adjust adversarial weights during training\n')

    bestEp=1

    for epoch in range(args.num_epochs):

        if epoch+1-bestEp>args.early_stop:
            print("Val Score hasn't improved for {} epochs. Early Stopping by {} epochs.".format(args.early_stop, args.num_epochs-epoch))
            break

        since = time.time()

        if adv_dict:
            model.set_LAMBDA(lambda_scheduler(epoch))

        if args.entropy_cond and not args.baseline:
            model, train_loss, train_score_dict = train_epoch_ec(model=model, training_generator=training_generator, loss_weights=loss_weights,
                                                                 loss_dict=loss_dict, adv_dict=adv_dict, optimizer=optimizer, device=device,
                                                                 zero_shot=args.zero_shot, prior_matrix=prior_matrix_tr, args=args)
        else:
            model, train_loss, train_score_dict = train_epoch(model=model, training_generator=training_generator, loss_weights=loss_weights,
                                                              loss_dict=loss_dict, optimizer=optimizer, device=device, zero_shot=args.zero_shot,
                                                              prior_matrix=prior_matrix_tr, args=args)

        model_wts_epoch = copy.deepcopy(model.state_dict())

        val_loss, val_score_dict = val_epoch(model=model, validation_generator=validation_generator, loss_dict=loss_dict,
                                             zero_shot=args.zero_shot, prior_matrix=prior_matrix_v, args=args, device=device)
        test_score_dict = test_model(model=model, test_generator=test_generator, loss_dict=loss_dict, zero_shot=args.zero_shot,
                                     prior_matrix=prior_matrix_ts, device=device)

        training_loss.append(train_loss)
        valid_loss.append(val_loss)

        if not adv_dict:
            lr_scheduler.step(val_loss)

        if args.zero_shot:
            lookup_key='acc'
            train_acc.append(train_score_dict[lookup_key])
            val_acc.append(val_score_dict[lookup_key])
            test_acc.append(test_score_dict[lookup_key])
        else:
            lookup_key='mAP'
            train_ap.append(train_score_dict[lookup_key])
            val_ap.append(val_score_dict[lookup_key])
            test_ap.append(test_score_dict[lookup_key])

        if val_score_dict[lookup_key]>best_val:
            best_val = val_score_dict[lookup_key]
            bestEp = epoch+1
            best_model_wts = model_wts_epoch

        time_elapsed = time.time() - since

        print('Epoch:{}/{}, Time Elapsed:{:.0f}m {:.0f}s'.format(epoch+1, args.num_epochs, time_elapsed // 60, time_elapsed % 60))
        print('Training loss:{}, Validation loss:{}; '.format(train_loss, val_loss))
        print('Train {}:{}; Validation {}:{}; Test {}:{}'.format(lookup_key, train_score_dict[lookup_key], lookup_key, val_score_dict[lookup_key],
                lookup_key, test_score_dict[lookup_key]))
        print('\n')

    print('Best performance at epoch {} with Validation {}:{}.'.format(bestEp, lookup_key, best_val))

    print('Testing with best model')
    model.load_state_dict(best_model_wts)
    test_score_dict = test_model(model=model, test_generator=test_generator, loss_dict=loss_dict, zero_shot=args.zero_shot,
                                 prior_matrix=prior_matrix_ts, device=device)
    print('Test {}:{}'.format(lookup_key, test_score_dict[lookup_key]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Grouped Adversarial Learning for Attribute prediction/Zero-Shot learning using images in aPY dataset.")
    parser.add_argument('-base', '--baseline', help="1 corressponds to a baseline with no adversarials", default=0, type=int)
    parser.add_argument('-zsl', '--zero_shot', help="if 1 the model does zero shot learning; if 2 or 3(indiv neurons), attr pred losses are also taken into account", default=1, type=int)
    parser.add_argument('-zsl_loss', '--zsl_loss_func', help="choose b/w eszsl, sje, devise, and ale", default='eszsl', type=str)
    parser.add_argument('-mr', '--margin', help="margin for ssvm based zsl loss functions (ale, sje, devise), active only if zsl_loss_func!='eszsl'", default=1, type=float)
    parser.add_argument('-l2', '--dense_reg', help="regularization weight for last dense layers (main and/or adv branches)", default=1e-5, type=float)
    parser.add_argument('-e', '--num_epochs', help="number of epochs to run", default=50, type=int)
    parser.add_argument('-es', '--early_stop', help="patience on val acc", default=20, type=int)
    parser.add_argument('-ie', '--increment_epochs', help="epoch no. till which adv_task_weight gets incremented from 0.0 to adv_task_weight, not in effect during baseline", default=-1, type=int)
    parser.add_argument('-bs', '--batch_size', help="batch size used for training", default=128, type=int)
    parser.add_argument('-lr', '--learning_rate', help="learning rate for adam/sgd", default=0.001, type=float)
    parser.add_argument('-cls', '--clusters', help="specify number of spectral grps (>1) - active only when gr_type is sp", default=3, type=int)
    parser.add_argument('-gr', '--groups', help="main tasks", default=['all'], nargs='+')
    parser.add_argument('-adv', '--adv_list', help="adv tasks", default=['all'], nargs='+')
    parser.add_argument('-lsb', '--latent_size_before', help='dense layer configurations before group heads', default=[], nargs='+')
    parser.add_argument('-lsa', '--latent_size_after', help='dense layer configurations for main (and adv) branches', default=[], nargs='+')
    parser.add_argument('-relu', '--add_relu', help='add non-linearity or not after linear layers', default=0, type=int)
    parser.add_argument('-drop', '--drop_rate', help='Dropout rate only for zero shot experiments', default=0.0, type=float)
    parser.add_argument('-ec', '--entropy_cond', help='specify whether or not to use entropy conditioning (non-baseline expt)', default=0, type=int)
    parser.add_argument('-grad_rev', '--grad_reversal_method', help='choose between vanilla grad reversal and grad projection, none operate during a baseline', default='neg-grad', type=str)
    parser.add_argument('-proj_v', '--proj_version', help='0 for neg-grad/baseline, 1 for proj then sum, 2 for sum then proj', default=0, type=int)
    parser.add_argument('-b', '--balance', help='if 1 the model uses weighted loss with weights inversely proportional to count in training for main (and adv) branches', default=1, type=int)
    parser.add_argument('-advb', '--adv_balance', help='if 1/2 the model uses weighted adv branches, 0 corressponds to unity weight for all branches', default=2, type=int)
    parser.add_argument('-advth', '--adv_threshold', help='0 for including all grps as adv branches, non-zero for discarding some', default=0.0, type=float)
    parser.add_argument('-opt', '--optimizer', help='if 0 the optimizer changes to SGD', default=0, type=int)
    parser.add_argument('-tw', '--main_task_weight', help='weight to give to main task', default=1., type=float)
    parser.add_argument('-aw', '--adv_task_weight', help='weight for adversarial tasks', default=1., type=float)
    parser.add_argument('-ia', '--individual_adv_weight', help='if 1 model uses different weights per individual adv tasks, 0 all equal', default=0., type=float)
    parser.add_argument('-seed', '--rand_seed', help='seed to set manual_seed in torch', default=4.0, type=float)
    parser.add_argument('-ds', '--data_split', help='data split to choose: "gbu" for GBU, "tv" for train+val, "cs" for correlation shift', default='gbu', type=str)

    args = parser.parse_args()
    print(args)
    main(args)
