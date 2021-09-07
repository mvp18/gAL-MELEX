import numpy as np
import torch
from torch.utils import data
from scipy import io
from collections import defaultdict

def load_data():

    res101 = io.loadmat('../../xlsa17/data/AWA2/res101.mat')
    att_splits = io.loadmat('../att_splits.mat') #ZSL_GBU data
    image_name_list = res101['image_files']
    binarized_predicates = io.loadmat('../binaryAtt_splits.mat')['att'].transpose()
    all_classes = [class_name[0][0] for class_name in att_splits['allclasses_names']]
    prior_matrix = att_splits['att'].transpose()

    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'

    image_names = np.array(['/'.join(y[0][0].split('/')[-2:]) for y in image_name_list])
    train_img_names = image_names[np.squeeze(att_splits[train_loc]-1)]
    val_img_names = image_names[np.squeeze(att_splits[val_loc]-1)]
    test_img_names = image_names[np.squeeze(att_splits[test_loc]-1)]

    X = res101['features']
    X = X.transpose()
    name2feat = dict(zip(image_names, X))

    class_labels = res101['labels']
    name2class = dict(zip(image_names, np.squeeze(class_labels)))

    class2att = dict(zip(all_classes, binarized_predicates))

    train_classes = class_labels[np.squeeze(att_splits[train_loc]-1)]
    val_classes = class_labels[np.squeeze(att_splits[val_loc]-1)]
    test_classes = class_labels[np.squeeze(att_splits[test_loc]-1)]

    prior_matrix_tr = prior_matrix[(np.unique(train_classes)-1),:]
    prior_matrix_val = prior_matrix[(np.unique(val_classes)-1),:]
    prior_matrix_ts = prior_matrix[(np.unique(test_classes)-1),:]

    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    img_name_list = [train_img_names, val_img_names, test_img_names]
    class_list = [train_classes, val_classes, test_classes]

    return all_classes, img_name_list, class_list, signature_list, name2feat, name2class, class2att

# New splits: will have train+val in train, train+val in val and test in test
def load_data_train_val_split():

    res101 = io.loadmat('../../xlsa17/data/AWA2/res101.mat')
    att_splits = io.loadmat('../att_splits.mat') #ZSL_GBU data
    image_name_list = res101['image_files']
    binarized_predicates = io.loadmat('../binaryAtt_splits.mat')['att'].transpose()
    all_classes = [class_name[0][0] for class_name in att_splits['allclasses_names']]
    prior_matrix = att_splits['att'].transpose()

    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'

    # Merging train and val ids. We will use the same fot val set.
    train_ids = np.append(np.squeeze(att_splits[train_loc]-1), np.squeeze(att_splits[val_loc]-1))
    test_ids = np.squeeze(att_splits[test_loc]-1)

    image_names = np.array(['/'.join(y[0][0].split('/')[-2:]) for y in image_name_list])

    train_img_names = image_names[train_ids]
    val_img_names = image_names[train_ids] # using same train_ids here.
    test_img_names = image_names[test_ids]

    X = res101['features']
    X = X.transpose()
    name2feat = dict(zip(image_names, X))

    class_labels = res101['labels']
    name2class = dict(zip(image_names, np.squeeze(class_labels)))

    class2att = dict(zip(all_classes, binarized_predicates))

    train_classes = class_labels[train_ids]
    val_classes = class_labels[train_ids]
    test_classes = class_labels[test_ids]

    prior_matrix_tr = prior_matrix[(np.unique(train_classes)-1),:]
    prior_matrix_val = prior_matrix[(np.unique(val_classes)-1),:]
    prior_matrix_ts = prior_matrix[(np.unique(test_classes)-1),:]

    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    img_name_list = [train_img_names, val_img_names, test_img_names]
    class_list = [train_classes, val_classes, test_classes]

    return all_classes, img_name_list, class_list, signature_list, name2feat, name2class, class2att

def load_data_custom_split(fp_class_splits):
    # fp_class_splits should be a link to npy file with dict with 'train_cls', 'test_cls' and 'val_cls' keys.
    cls_splits = np.load(fp_class_splits, allow_pickle=True).item()
    
    res101 = io.loadmat('../../xlsa17/data/AWA2/res101.mat')
    att_splits = io.loadmat('../att_splits.mat') #ZSL_GBU data
    image_name_list = res101['image_files']
    binarized_predicates = io.loadmat('../binaryAtt_splits.mat')['att'].transpose()
    all_classes = [class_name[0][0] for class_name in att_splits['allclasses_names']]
    prior_matrix = att_splits['att'].transpose()
    image_names = np.array(['/'.join(y[0][0].split('/')[-2:]) for y in image_name_list])
    
    train_loc = []
    val_loc = []
    test_loc = []
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

    X = res101['features']
    X = X.transpose()
    name2feat = dict(zip(image_names, X))

    class_labels = res101['labels']
    name2class = dict(zip(image_names, np.squeeze(class_labels)))

    class2att = dict(zip(all_classes, binarized_predicates))

    train_classes = class_labels[np.squeeze(train_loc)]
    val_classes = class_labels[np.squeeze(val_loc)]
    test_classes = class_labels[np.squeeze(test_loc)]

    prior_matrix_tr = prior_matrix[(np.unique(train_classes)-1),:]
    prior_matrix_val = prior_matrix[(np.unique(val_classes)-1),:]
    prior_matrix_ts = prior_matrix[(np.unique(test_classes)-1),:]

    signature_list = [prior_matrix_tr, prior_matrix_val, prior_matrix_ts]
    img_name_list = [train_img_names, val_img_names, test_img_names]
    class_list = [train_classes, val_classes, test_classes]

    return all_classes, img_name_list, class_list, signature_list, name2feat, name2class, class2att

class Dataset(data.Dataset):
    def __init__(self, list_IDs, data_dict, class2att, class_dict, eszsl_classes, all_classes, attributes, attribute_groups, groups, adv_dict=[], zero_shot=0):
        
        self.data_dict = data_dict
        self.class2att = class2att
        self.class_dict = class_dict
        self.all_classes = all_classes
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

            sample_class = self.class_dict[ID]
            attr_labels = self.class2att[self.all_classes[sample_class-1]]
            
            feed_dict={}
            
            for group in self.groups:
                att_indices_per_group = [self.attributes.index(att) for att in self.attribute_groups[group]]
                feed_dict[group] = np.array(attr_labels)[att_indices_per_group]
            
            return dict([(k, np.array(v, dtype=np.float32)) for k, v in feed_dict.items()])
            
        # Generate data

        if self.zero_shot:
            group_dict = _create_feed()
            if self.zero_shot==2 or self.zero_shot==3:
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
