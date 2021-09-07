import numpy as np
import torch
from torch.utils import data
from scipy import io
from collections import defaultdict
import pickle

def load_data():
    
    res101 = io.loadmat('../../xlsa17/data/APY/res101.mat')
    att_splits = io.loadmat('../att_splits.mat')
    prior_matrix = att_splits['att']
    allclass_names=att_splits['allclasses_names']

    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'

    X = res101['features']
    X = X.transpose()

    train_X = X[np.squeeze(att_splits[train_loc]-1)]
    val_X = X[np.squeeze(att_splits[val_loc]-1)]
    test_X = X[np.squeeze(att_splits[test_loc]-1)]

    all_classes = res101['labels']
    train_classes = np.squeeze(all_classes[np.squeeze(att_splits[train_loc]-1)])
    val_classes = np.squeeze(all_classes[np.squeeze(att_splits[val_loc]-1)])
    test_classes = np.squeeze(all_classes[np.squeeze(att_splits[test_loc]-1)])

    prior_matrix = att_splits['att']
    prior_matrix_tr = prior_matrix[:,(np.unique(train_classes)-1)]
    prior_matrix_val = prior_matrix[:,(np.unique(val_classes)-1)]
    prior_matrix_ts = prior_matrix[:,(np.unique(test_classes)-1)]

    train_img=res101['image_files'][np.squeeze(att_splits[train_loc]-1)]
    val_img=res101['image_files'][np.squeeze(att_splits[val_loc]-1)]
    test_img=res101['image_files'][np.squeeze(att_splits[test_loc]-1)]
    
    train_img_names=[]
    val_img_names=[]
    test_img_names=[]
    for i in range(train_img.shape[0]):
        train_img_names.append(train_img[i][0][0].split('ages/')[1]+'+'+allclass_names[train_classes[i]-1][0][0])
    for i in range(test_img.shape[0]):
        test_img_names.append(test_img[i][0][0].split('ages/')[1]+'+'+allclass_names[test_classes[i]-1][0][0])
    for i in range(val_img.shape[0]):
        val_img_names.append(val_img[i][0][0].split('ages/')[1]+'+'+allclass_names[val_classes[i]-1][0][0])

    all_img_names=train_img_names+val_img_names+test_img_names
    rept_img_dict={}
    for i in range(len(all_img_names)):
        rept_img_dict[all_img_names[i]]=0

    img2att={}
    
    apascal_train=open('../attribute_data/apascal_train.txt').readlines()
    apascal_test=open('../attribute_data/apascal_test.txt').readlines()
    ayahoo_test=open('../attribute_data/ayahoo_test.txt').readlines()
    
    for i in range(len(apascal_train)):
        img_name=apascal_train[i].split(' ')[0]+'+'+apascal_train[i].split(' ')[1]
        if all_img_names.count(img_name)>1:
            suffix=str(rept_img_dict[img_name]+1)
            rept_img_dict[img_name]+=1
        else:
            suffix='1'
        bin_att=[]
        for j in range(64):
            bin_att.append(float(apascal_train[i].split(' ')[j+6]))
        img2att[img_name+'_'+suffix]=np.array(bin_att)
    
    for a in range(len(apascal_test)):
        img_name=apascal_test[a].split(' ')[0]+'+'+apascal_test[a].split(' ')[1]
        if all_img_names.count(img_name)>1:
            suffix=str(rept_img_dict[img_name]+1)
            rept_img_dict[img_name]+=1
        else:
            suffix='1'
        bin_att=[]
        for b in range(64):
            bin_att.append(float(apascal_test[a].split(' ')[b+6]))
        img2att[img_name+'_'+suffix]=np.array(bin_att)
    
    for x in range(len(ayahoo_test)):
        img_name=ayahoo_test[x].split(' ')[0]+'+'+ayahoo_test[x].split(' ')[1]
        if all_img_names.count(img_name)>1:
            suffix=str(rept_img_dict[img_name]+1)
            rept_img_dict[img_name]+=1
        else:
            suffix='1'
        bin_att=[]
        for y in range(64):
            bin_att.append(float(ayahoo_test[x].split(' ')[y+6]))
        img2att[img_name+'_'+suffix]=np.array(bin_att)

    train_att={}
    for img_name in set(train_img_names):
        if rept_img_dict[img_name]!=0:
            for j in range(rept_img_dict[img_name]):
                if j:
                    temp_arr=np.vstack([temp_arr, np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)])
                else:
                    temp_arr=np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)
            train_att[img_name]=np.round(np.mean(temp_arr,0))
        else:
            train_att[img_name]=img2att[img_name+'_1']
    
    val_att={}
    for img_name in set(val_img_names):
        if rept_img_dict[img_name]!=0:
            for j in range(rept_img_dict[img_name]):
                if j:
                    temp_arr=np.vstack([temp_arr, np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)])
                else:
                    temp_arr=np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)
            val_att[img_name]=np.round(np.mean(temp_arr,0))
        else:
            val_att[img_name]=img2att[img_name+'_1']
    
    test_att={}
    for img_name in set(test_img_names):
        if rept_img_dict[img_name]!=0:
            for j in range(rept_img_dict[img_name]):
                if j:
                    temp_arr=np.vstack([temp_arr, np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)])
                else:
                    temp_arr=np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)
            test_att[img_name]=np.round(np.mean(temp_arr,0))
        else:
            test_att[img_name]=img2att[img_name+'_1']

    img_name_list=[train_img_names, val_img_names, test_img_names]
    feature_list = [train_X, val_X, test_X]
    class_list = [train_classes, val_classes, test_classes]
    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    att_list=[train_att, val_att, test_att]

    return img_name_list, class_list, signature_list, feature_list, att_list, img2att, rept_img_dict 

def load_data_custom_split(fp_class_splits):
    
    cls_splits = np.load(fp_class_splits, allow_pickle=True).item()
    
    res101 = io.loadmat('../../xlsa17/data/APY/res101.mat')
    att_splits = io.loadmat('../att_splits.mat')
    prior_matrix = att_splits['att']

    all_classes = res101['labels']    
    allclass_names=att_splits['allclasses_names']

    train_loc = []
    val_loc = []
    test_loc = []
    
    for i, label in enumerate(all_classes):
        if allclass_names[label-1] in cls_splits['train_cls']:
            train_loc.append(i)
        elif allclass_names[label-1] in cls_splits['val_cls']:
            val_loc.append(i)
        elif allclass_names[label-1] in cls_splits['test_cls']:
            test_loc.append(i)
            
    X = res101['features']
    X = X.transpose()

    train_X = X[np.squeeze(train_loc)]
    val_X = X[np.squeeze(val_loc)]
    test_X = X[np.squeeze(test_loc)]


    train_classes = np.squeeze(all_classes[np.squeeze(train_loc)])
    val_classes = np.squeeze(all_classes[np.squeeze(val_loc)])
    test_classes = np.squeeze(all_classes[np.squeeze(test_loc)])

    prior_matrix = att_splits['att']
    prior_matrix_tr = prior_matrix[:,(np.unique(train_classes)-1)]
    prior_matrix_val = prior_matrix[:,(np.unique(val_classes)-1)]
    prior_matrix_ts = prior_matrix[:,(np.unique(test_classes)-1)]

    train_img=res101['image_files'][np.squeeze(train_loc)]
    val_img=res101['image_files'][np.squeeze(val_loc)]
    test_img=res101['image_files'][np.squeeze(test_loc)]
    
    train_img_names=[]
    val_img_names=[]
    test_img_names=[]
    for i in range(train_img.shape[0]):
        train_img_names.append(train_img[i][0][0].split('ages/')[1]+'+'+allclass_names[train_classes[i]-1][0][0])
    for i in range(test_img.shape[0]):
        test_img_names.append(test_img[i][0][0].split('ages/')[1]+'+'+allclass_names[test_classes[i]-1][0][0])
    for i in range(val_img.shape[0]):
        val_img_names.append(val_img[i][0][0].split('ages/')[1]+'+'+allclass_names[val_classes[i]-1][0][0])

    all_img_names=train_img_names+val_img_names+test_img_names
    rept_img_dict={}
    for i in range(len(all_img_names)):
        rept_img_dict[all_img_names[i]]=0

    img2att={}
    
    apascal_train=open('../attribute_data/apascal_train.txt').readlines()
    apascal_test=open('../attribute_data/apascal_test.txt').readlines()
    ayahoo_test=open('../attribute_data/ayahoo_test.txt').readlines()
    
    for i in range(len(apascal_train)):
        img_name=apascal_train[i].split(' ')[0]+'+'+apascal_train[i].split(' ')[1]
        if all_img_names.count(img_name)>1:
            suffix=str(rept_img_dict[img_name]+1)
            rept_img_dict[img_name]+=1
        else:
            suffix='1'
        bin_att=[]
        for j in range(64):
            bin_att.append(float(apascal_train[i].split(' ')[j+6]))
        img2att[img_name+'_'+suffix]=np.array(bin_att)
    
    for a in range(len(apascal_test)):
        img_name=apascal_test[a].split(' ')[0]+'+'+apascal_test[a].split(' ')[1]
        if all_img_names.count(img_name)>1:
            suffix=str(rept_img_dict[img_name]+1)
            rept_img_dict[img_name]+=1
        else:
            suffix='1'
        bin_att=[]
        for b in range(64):
            bin_att.append(float(apascal_test[a].split(' ')[b+6]))
        img2att[img_name+'_'+suffix]=np.array(bin_att)
    
    for x in range(len(ayahoo_test)):
        img_name=ayahoo_test[x].split(' ')[0]+'+'+ayahoo_test[x].split(' ')[1]
        if all_img_names.count(img_name)>1:
            suffix=str(rept_img_dict[img_name]+1)
            rept_img_dict[img_name]+=1
        else:
            suffix='1'
        bin_att=[]
        for y in range(64):
            bin_att.append(float(ayahoo_test[x].split(' ')[y+6]))
        img2att[img_name+'_'+suffix]=np.array(bin_att)

    train_att={}
    for img_name in set(train_img_names):
        if rept_img_dict[img_name]!=0:
            for j in range(rept_img_dict[img_name]):
                if j:
                    temp_arr=np.vstack([temp_arr, np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)])
                else:
                    temp_arr=np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)
            train_att[img_name]=np.round(np.mean(temp_arr,0))
        else:
            train_att[img_name]=img2att[img_name+'_1']
    
    val_att={}
    for img_name in set(val_img_names):
        if rept_img_dict[img_name]!=0:
            for j in range(rept_img_dict[img_name]):
                if j:
                    temp_arr=np.vstack([temp_arr, np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)])
                else:
                    temp_arr=np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)
            val_att[img_name]=np.round(np.mean(temp_arr,0))
        else:
            val_att[img_name]=img2att[img_name+'_1']
    
    test_att={}
    for img_name in set(test_img_names):
        if rept_img_dict[img_name]!=0:
            for j in range(rept_img_dict[img_name]):
                if j:
                    temp_arr=np.vstack([temp_arr, np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)])
                else:
                    temp_arr=np.expand_dims(img2att[img_name+'_'+str(j+1)], 0)
            test_att[img_name]=np.round(np.mean(temp_arr,0))
        else:
            test_att[img_name]=img2att[img_name+'_1']

    img_name_list=[train_img_names, val_img_names, test_img_names]
    feature_list = [train_X, val_X, test_X]
    class_list = [train_classes, val_classes, test_classes]
    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    att_list=[train_att, val_att, test_att]

    return img_name_list, class_list, signature_list, feature_list, att_list, img2att, rept_img_dict 

class Dataset(data.Dataset):
    def __init__(self, list_IDs, data_dict, labels_dict, class_list, eszsl_classes, attributes, attribute_groups, groups, adv_dict=[], zero_shot=0):
        
        self.data_dict = data_dict
        self.labels_dict = labels_dict
        self.class_list = class_list
        self.eszsl_classes = eszsl_classes
        self.list_IDs = list_IDs
        self.attributes = attributes
        self.attribute_groups = attribute_groups
        self.groups = groups
        self.adv_dict = adv_dict
        self.zero_shot = zero_shot
  
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        X, y = self.__data_generation(ID, index)

        return X, y

    def __data_generation(self, ID, index):

        def _create_feed():

            label = self.labels_dict[ID]
            
            feed_dict={}
            
            for group in self.groups:
                att_indices_per_group = [self.attributes.index(att) for att in self.attribute_groups[group]]
                feed_dict[group] = np.array(label)[att_indices_per_group]
            
            return dict([(k, np.array(v, dtype=np.float32)) for k, v in feed_dict.items()])
            
        # Generate data

        if self.zero_shot:
            group_dict = _create_feed()
            if self.zero_shot==2:
                y_dict = group_dict
            else:
                y_dict={}
            y_dict['conc_l'] = self.eszsl_classes.index(self.class_list[index])
            if self.adv_dict:
                for adv_branch in self.adv_dict:
                    y_dict[adv_branch['node_name']] = group_dict[adv_branch['group']]
        else:
            y_dict = _create_feed()
            if self.adv_dict:
                for adv_branch in self.adv_dict: 
                    y_dict[adv_branch['node_name']] = y_dict[adv_branch['group']]

        X = self.data_dict[index]
        X = torch.from_numpy(X)
        
        return X, y_dict
    
