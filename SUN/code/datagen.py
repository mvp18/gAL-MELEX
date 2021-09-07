import numpy as np
import torch
from torch.utils import data
from scipy import io
import pickle

def load_data():

    res101 = io.loadmat('../../xlsa17/data/SUN/res101.mat')
    att_splits = io.loadmat('../att_splits.mat') #ZSL_GBU data
    image_names = np.array([x[0][0].split('images/')[1] for x in res101['image_files']])

    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'
    
    train_img_names = image_names[np.squeeze(att_splits[train_loc]-1)]
    val_img_names = image_names[np.squeeze(att_splits[val_loc]-1)]
    test_img_names = image_names[np.squeeze(att_splits[test_loc]-1)]
    
    X = res101['features']
    X = X.transpose()
    name2feat = dict(zip(image_names, X))

    class_labels = res101['labels']
    name2class = dict(zip(image_names, np.squeeze(class_labels)))

    train_classes = class_labels[np.squeeze(att_splits[train_loc]-1)]
    val_classes = class_labels[np.squeeze(att_splits[val_loc]-1)]
    test_classes = class_labels[np.squeeze(att_splits[test_loc]-1)]

    prior_matrix = att_splits['att']
    prior_matrix_tr = prior_matrix[:, (np.unique(train_classes)-1)]
    prior_matrix_val = prior_matrix[:, (np.unique(val_classes)-1)]
    prior_matrix_ts = prior_matrix[:, (np.unique(test_classes)-1)]

    image_names = [x[0][0] for x in io.loadmat('../images.mat')['images']]
    image_attributes = np.round(io.loadmat('../attributeLabels_continuous.mat')['labels_cv'])
    name2att = dict(zip(image_names, image_attributes))

    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    img_name_list = [train_img_names, val_img_names, test_img_names]
    class_list = [train_classes, val_classes, test_classes]

    return img_name_list, class_list, signature_list, name2feat, name2class, name2att


def load_data_custom_split(fp_class_splits):

    cls_splits = np.load(fp_class_splits, allow_pickle=True).item()

    res101 = io.loadmat('../../xlsa17/data/SUN/res101.mat')
    att_splits = io.loadmat('../att_splits.mat') #ZSL_GBU data
    image_names = np.array([x[0][0].split('images/')[1] for x in res101['image_files']])
    class_labels = res101['labels']

    train_loc = []
    val_loc = []
    test_loc = []
    
    for i, label in enumerate(class_labels):
        if label in cls_splits['train_cls']:
            train_loc.append(i)
        elif label in cls_splits['val_cls']:
            val_loc.append(i)
        elif label in cls_splits['test_cls']:
            test_loc.append(i)
    
    train_img_names = image_names[np.squeeze(train_loc)]
    val_img_names = image_names[np.squeeze(val_loc)]
    test_img_names = image_names[np.squeeze(test_loc)]
    
    X = res101['features']
    X = X.transpose()
    name2feat = dict(zip(image_names, X))

    name2class = dict(zip(image_names, np.squeeze(class_labels)))

    train_classes = class_labels[np.squeeze(train_loc)]
    val_classes = class_labels[np.squeeze(val_loc)]
    test_classes = class_labels[np.squeeze(test_loc)]

    prior_matrix = att_splits['att']
    prior_matrix_tr = prior_matrix[:, (np.unique(train_classes)-1)]
    prior_matrix_val = prior_matrix[:, (np.unique(val_classes)-1)]
    prior_matrix_ts = prior_matrix[:, (np.unique(test_classes)-1)]

    image_names = [x[0][0] for x in io.loadmat('../images.mat')['images']]
    image_attributes = np.round(io.loadmat('../attributeLabels_continuous.mat')['labels_cv'])
    name2att = dict(zip(image_names, image_attributes))

    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    img_name_list = [train_img_names, val_img_names, test_img_names]
    class_list = [train_classes, val_classes, test_classes]

    return img_name_list, class_list, signature_list, name2feat, name2class, name2att


class Dataset(data.Dataset):
    def __init__(self, list_IDs, data_dict, name2att, class_dict, eszsl_classes, attributes, attribute_groups, groups, adv_dict=[], zero_shot=0):
        
        self.data_dict = data_dict
        self.name2att = name2att
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

            attr_labels = self.name2att[ID]
            
            feed_dict={}
            
            for group in self.groups:
                att_indices_per_group = [self.attributes.index(att) for att in self.attribute_groups[group]]
                feed_dict[group] = np.array(attr_labels)[att_indices_per_group]
            
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
