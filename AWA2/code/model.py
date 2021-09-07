import torch
import torch.nn as nn
from torch.autograd import Function

class GradReverse(Function):    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        reversed_scaled_grad = torch.neg(ctx.lambda_ * grad_output.clone())
        return reversed_scaled_grad, None

def grad_reverse(x, LAMBDA):
    return GradReverse.apply(x, LAMBDA)

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res

class GAL(nn.Module):
    def __init__(self, predicate_groups=None, groups=None, adv_predicate_groups=None, vector_size=[], proj_version=0,
                 ec=0, adv_dict=[], loss_weights=None, zero_shot=0, drop_rate=0.0, add_relu=True):
        super(GAL, self).__init__()

        self.predicate_groups = predicate_groups
        self.groups = groups
        self.adv_dict = adv_dict
        self.vsize = vector_size
        self.drop_rate = drop_rate
        self.add_relu = add_relu
        self.entropy_cond = ec
        self.zero_shot = zero_shot

        if adv_predicate_groups:
            self.adv_predicate_groups = adv_predicate_groups
        else:
            self.adv_predicate_groups = predicate_groups.copy()

        self.input_layer_dim = 2048

        if isinstance(self.vsize[0], list) and len(self.vsize[0]) > 0:
            # vector_size is list of lists. Using first list now
            lsb = []
            for i in range(len(self.vsize[0])):
                if i:
                    lsb.append(self._internal_linear_block(self.vsize[0][i-1], self.vsize[0][i]))
                else:
                    lsb.append(self._internal_linear_block(self.input_layer_dim, self.vsize[0][i]))
            self.lsb = nn.ModuleList(lsb)
        else:
            self.lsb = None

        self.group_heads = nn.ModuleDict(self._create_group_heads())
  
        if isinstance(self.vsize[1], list) and len(self.vsize[1]) > 1:
            lsa_main_dict = {}
            for group in self.groups:
                lsa_main_dict[group] = self._build_lsa_backbones()
            self.lsa_main_dict = nn.ModuleDict(lsa_main_dict)    
        else:
            self.lsa_main_dict = None
        
        self.group_class_scores = nn.ModuleDict(self._build_main_gr_extensions())
        
        if self.adv_dict:
            if isinstance(self.vsize[1], list) and len(self.vsize[1]) > 1:
                lsa_adv_dict = {}
                for adv_branch in self.adv_dict:
                    lsa_adv_dict[adv_branch['node_name']] = self._build_lsa_backbones()
                self.lsa_adv_dict = nn.ModuleDict(lsa_adv_dict)
            else:
                self.lsa_adv_dict = None
            self.adv_class_scores = nn.ModuleDict(self._build_adv_extensions())

        if self.zero_shot==3:
            self.att_pred_eszsl3 = nn.ModuleDict(self._build_eszsl3_att_predictors())

    def set_LAMBDA(self, LAMBDA):
        self.LAMBDA = LAMBDA
        print('LAMBDA set to {}.'.format(self.LAMBDA))

    def _internal_linear_block(self, in_feat, out_feat):
        
        linear_block = []
        linear_block.append(nn.Linear(in_feat, out_feat))
        linear_block.append(nn.BatchNorm1d(out_feat))
        if self.add_relu: linear_block.append(nn.LeakyReLU(0.01))
        linear_block.append(nn.Dropout(self.drop_rate))
        
        return nn.Sequential(*linear_block)

    def _classifier_linear_block(self, in_feat, out_feat, adv_flag):
        
        if adv_flag:
            linear_block = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.Sigmoid()
            )
        else:
            linear_block = nn.Linear(in_feat, out_feat)
        
        return linear_block
    
    def _build_lsa_backbones(self):

        lsa = []
        for i in range(1, len(self.vsize[1])):
            lsa.append(self._internal_linear_block(self.vsize[1][i-1], self.vsize[1][i]))
        
        return nn.ModuleList(lsa)

    def _create_group_heads(self):

        gr_head_dict = {}
        for group in self.groups:
            if isinstance(self.vsize[1], list) and len(self.vsize[1])>0:
                if self.lsb:
                    gr_head_dict['latent_'+group] = self._internal_linear_block(self.vsize[0][-1], self.vsize[1][0])
                else:
                    gr_head_dict['latent_'+group] = self._internal_linear_block(self.input_layer_dim, self.vsize[1][0])
        return gr_head_dict

    def _build_main_gr_extensions(self):
        
        main_gr_class_scores = {}
        for gr in self.groups:
            main_gr_class_scores[gr] = self._classifier_linear_block(self.vsize[1][-1], len(self.predicate_groups[gr]), 0)

        return main_gr_class_scores

    def _build_adv_extensions(self):

        adv_gr_class_scores = {}

        for adv_branch in self.adv_dict:
            adv_gr_class_scores[adv_branch['node_name']] = self._classifier_linear_block(self.vsize[1][-1], len(self.adv_predicate_groups[adv_branch['group']]), 1)

        return adv_gr_class_scores

    def _build_eszsl3_att_predictors(self):

        att_pred_eszsl3={}

        for gr in self.groups:
            indiv_att_neurons=[]
            for i in range(len(self.predicate_groups[gr])):
                indiv_att_neurons.append(self._classifier_linear_block(1,1,0))
            att_pred_eszsl3[gr]=nn.ModuleList(indiv_att_neurons)

        return att_pred_eszsl3
    
    def forward(self, x):
        
        if self.lsb:
            for linear_block in self.lsb:
                x = linear_block(x)
        
        gr_head_dict = {}
        main_gr_class_scores = {}
        adv_gr_class_scores = {}
        flip = {}

        if self.zero_shot==3:
            att_pred={}
        
        for gr in self.groups:
            gr_head_dict['latent_'+gr] = self.group_heads['latent_'+gr](x)
            if self.lsa_main_dict:
                gr_lsa = self.lsa_main_dict[gr][0](gr_head_dict['latent_'+gr])
                if len(self.lsa_main_dict[gr])>1:
                    for linear_block in self.lsa_main_dict[gr][1:]:
                        gr_lsa = linear_block(gr_lsa)
            else:
                gr_lsa = gr_head_dict['latent_'+gr]
            main_gr_class_scores[gr] = self.group_class_scores[gr](gr_lsa)

            if self.zero_shot==3:
                gr_class_score = main_gr_class_scores[gr]
                pred = torch.zeros_like(gr_class_score)
                for i, neuron in enumerate(self.att_pred_eszsl3[gr]):
                    pred[:, i] = torch.squeeze(neuron(torch.unsqueeze(gr_class_score[:, i], 1)), 1)
                att_pred[gr] = pred

        if self.adv_dict:
            for adv_branch in self.adv_dict:
                flip[adv_branch['node_name']] = grad_reverse(gr_head_dict[adv_branch['parent']], self.LAMBDA)
                if self.lsa_adv_dict:
                    adv_lsa = self.lsa_adv_dict[adv_branch['node_name']][0](flip[adv_branch['node_name']])
                    if len(self.lsa_adv_dict[adv_branch['node_name']])>1:
                        for linear_block in self.lsa_adv_dict[adv_branch['node_name']][1:]:
                            adv_lsa = linear_block(adv_lsa)
                else:
                    adv_lsa = flip[adv_branch['node_name']]
                adv_gr_class_scores[adv_branch['node_name']] = self.adv_class_scores[adv_branch['node_name']](adv_lsa)

            if self.zero_shot:
                eszsl_pred = {}
                eszsl_pred['conc_l'] = torch.cat(list(main_gr_class_scores.values()), dim=1)
                model_output = Merge(eszsl_pred, adv_gr_class_scores)
                if self.entropy_cond or self.zero_shot==2:
                    model_output = Merge(model_output, main_gr_class_scores)
                if self.zero_shot==3: # Not to be used with entropy_cond, wd result in mixup, fix later
                    model_output = Merge(model_output, att_pred)
            else:
                model_output = Merge(main_gr_class_scores, adv_gr_class_scores)
        else:
            if self.zero_shot:
                eszsl_pred = {}
                eszsl_pred['conc_l'] = torch.cat(list(main_gr_class_scores.values()), dim=1)
                model_output = eszsl_pred
            else:
                model_output = main_gr_class_scores
        
        return model_output 
