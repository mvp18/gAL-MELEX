import numpy as np
import torch
from torch.utils import data
from scipy import io
from collections import defaultdict

def load_data():

    res101 = io.loadmat('../../xlsa17/data/CUB/res101.mat')
    att_splits = io.loadmat('../att_splits.mat')
    image_name_list = res101['image_files']
    prior_matrix = att_splits['att'].transpose()

    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'

    image_names = np.array(['/'.join(y[0][0].split('/')[-2:]) for y in image_name_list])
    train_img_names = image_names[np.squeeze(att_splits[train_loc]-1)]
    val_img_names = image_names[np.squeeze(att_splits[val_loc]-1)]
    test_img_names = image_names[np.squeeze(att_splits[test_loc]-1)]
    
    name_id_path = '../images.txt'
    name_id = open(name_id_path).read().splitlines()

    id2name = {}
    for img in name_id:
        id2name[img.split(' ')[0]] = img.split(' ')[1]
    
    imgattr_labels = open('../attributes/image_attribute_labels.txt', 'r')
    imgattr_labels = imgattr_labels.readlines()

    attr_dict = {}
    for i,line in enumerate(imgattr_labels):
        x=line.split(' ')
        if i%312==0:
            sample_labels = []
            sample_labels.append(int(x[2]))
        else:
            sample_labels.append(int(x[2]))
        if i!=0 and (i+1)%312==0:
            attr_dict[id2name[x[0]]] = np.array(sample_labels)

    X = res101['features']
    X = X.transpose()

    feature_lookup = dict(zip(image_names, X))

    all_classes = res101['labels']
    class_lookup = dict(zip(image_names, np.squeeze(all_classes)))

    train_classes = all_classes[np.squeeze(att_splits[train_loc]-1)]
    val_classes = all_classes[np.squeeze(att_splits[val_loc]-1)]
    test_classes = all_classes[np.squeeze(att_splits[test_loc]-1)]

    prior_matrix_tr = prior_matrix[(np.unique(train_classes)-1),:]
    prior_matrix_val = prior_matrix[(np.unique(val_classes)-1),:]
    prior_matrix_ts = prior_matrix[(np.unique(test_classes)-1),:]

    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    img_name_list = [train_img_names, val_img_names, test_img_names]
    class_list = [train_classes, val_classes, test_classes]

    return img_name_list, class_list, signature_list, feature_lookup, attr_dict, class_lookup 

def load_data_custom_split(fp_class_splits):
    # fp_class_splits should be a link to npy file with dict with 'train_cls', 'test_cls' and 'val_cls' keys.
    cls_splits = np.load(fp_class_splits, allow_pickle=True).item()

    res101 = io.loadmat('../../xlsa17/data/CUB/res101.mat')
    att_splits = io.loadmat('../att_splits.mat')
    image_name_list = res101['image_files']
    prior_matrix = att_splits['att'].transpose()

    train_loc = []
    val_loc = []
    test_loc = []

    image_names = np.array(['/'.join(y[0][0].split('/')[-2:]) for y in image_name_list])

    for i, name in enumerate(image_names):
        cls_name = name.split('/')[0]
        if cls_name in cls_splits['train_cls']:
            train_loc.append(i)
        elif cls_name in cls_splits['val_cls']:
            val_loc.append(i)
        else:
            test_loc.append(i)

    train_img_names = image_names[np.squeeze(train_loc)]
    val_img_names = image_names[np.squeeze(val_loc)]
    test_img_names = image_names[np.squeeze(test_loc)]
    
    name_id_path = '../images.txt'
    name_id = open(name_id_path).read().splitlines()

    id2name = {}
    for img in name_id:
        id2name[img.split(' ')[0]] = img.split(' ')[1]
    
    imgattr_labels = open('../attributes/image_attribute_labels.txt', 'r')
    imgattr_labels = imgattr_labels.readlines()

    attr_dict = {}
    for i,line in enumerate(imgattr_labels):
        x=line.split(' ')
        if i%312==0:
            sample_labels = []
            sample_labels.append(int(x[2]))
        else:
            sample_labels.append(int(x[2]))
        if i!=0 and (i+1)%312==0:
            attr_dict[id2name[x[0]]] = np.array(sample_labels)

    X = res101['features']
    X = X.transpose()

    feature_lookup = dict(zip(image_names, X))

    all_classes = res101['labels']
    class_lookup = dict(zip(image_names, np.squeeze(all_classes)))

    train_classes = all_classes[np.squeeze(train_loc)]
    val_classes = all_classes[np.squeeze(val_loc)]
    test_classes = all_classes[np.squeeze(test_loc)]

    prior_matrix_tr = prior_matrix[(np.unique(train_classes)-1),:]
    prior_matrix_val = prior_matrix[(np.unique(val_classes)-1),:]
    prior_matrix_ts = prior_matrix[(np.unique(test_classes)-1),:]

    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    img_name_list = [train_img_names, val_img_names, test_img_names]
    class_list = [train_classes, val_classes, test_classes]

    return img_name_list, class_list, signature_list, feature_lookup, attr_dict, class_lookup 

class Dataset(data.Dataset):
    def __init__(self, list_IDs, data_dict, labels_dict, class_dict, eszsl_classes, attributes, attribute_groups, groups, adv_dict=[], zero_shot=0):
        
        self.data_dict = data_dict
        self.labels_dict = labels_dict
        self.class_dict = class_dict
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

        X, y = self.__data_generation(ID)

        return X, y

    def __data_generation(self, ID):

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
            y_dict['conc_l'] = self.eszsl_classes.index(self.class_dict[ID])
            if self.adv_dict:
                for adv_branch in self.adv_dict:
                    y_dict[adv_branch['node_name']] = group_dict[adv_branch['group']]
        else:
            y_dict = _create_feed()
            if self.adv_dict:
                for adv_branch in self.adv_dict: 
                    y_dict[adv_branch['node_name']] = y_dict[adv_branch['group']]

        X = self.data_dict[ID]
        X = torch.from_numpy(X)
        
        return X, y_dict
    
