from __future__ import print_function
import argparse
import pickle
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
import sys
import os
import scipy
import random
cwd = os.getcwd()
sys.path.append(cwd+'/../')

def create_scaling_mat_ip_thres_bias(weight, ind, threshold, model_type):
    '''
    weight - 2D matrix (n_{i+1}, n_i), np.ndarray
    ind - chosen indices to remain, np.ndarray
    threshold - cosine similarity threshold
    '''
    assert(type(weight) == np.ndarray)
    assert(type(ind) == np.ndarray)

    cosine_sim = 1-pairwise_distances(weight, metric="cosine")
    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    for i in range(weight.shape[0]):
        if i in ind: # chosen
            ind_i, = np.where(ind == i)
            assert(len(ind_i) == 1) # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else: # not chosen
            if model_type == 'prune':
                continue
            max_cos_value = np.max(cosine_sim[i][ind])
            max_cos_value_index = np.argpartition(cosine_sim[i][ind], -1)[-1]

            if threshold and max_cos_value < threshold:
                continue

            baseline_weight = weight_chosen[max_cos_value_index]
            current_weight = weight[i]
            baseline_norm = np.linalg.norm(baseline_weight)
            current_norm = np.linalg.norm(current_weight)
            scaling_factor = current_norm / baseline_norm
            scaling_mat[i, max_cos_value_index] = scaling_factor

    return scaling_mat




def create_scaling_mat_conv_thres_bn(weight, ind, threshold,
                                     bn_weight, bn_bias,
                                     bn_mean, bn_var, lam, model_type):
    '''
    weight - 4D tensor(n, c, h, w), np.ndarray
    ind - chosen indices to remain
    threshold - cosine similarity threshold
    bn_weight, bn_bias - parameters of batch norm layer right after the conv layer
    bn_mean, bn_var - running_mean, running_var of BN (for inference)
    lam - how much to consider cosine sim over bias, float value between 0 and 1
    '''
    assert(type(weight) == np.ndarray)
    assert(type(ind) == np.ndarray)
    assert(type(bn_weight) == np.ndarray)
    assert(type(bn_bias) == np.ndarray)
    assert(type(bn_mean) == np.ndarray)
    assert(type(bn_var) == np.ndarray)
    assert(bn_weight.shape[0] == weight.shape[0])
    assert(bn_bias.shape[0] == weight.shape[0])
    assert(bn_mean.shape[0] == weight.shape[0])
    assert(bn_var.shape[0] == weight.shape[0])
    
    
    weight = weight.reshape(weight.shape[0], -1)

    cosine_dist = pairwise_distances(weight, metric="cosine")

    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    for i in range(weight.shape[0]):
        if i in ind: # chosen
            ind_i, = np.where(ind == i)
            assert(len(ind_i) == 1) # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else: # not chosen

            if model_type == 'prune':
                continue

            current_weight = weight[i]
            current_norm = np.linalg.norm(current_weight)
            current_cos = cosine_dist[i]
            gamma_1 = bn_weight[i]
            beta_1 = bn_bias[i]
            mu_1 = bn_mean[i]
            sigma_1 = bn_var[i]
            
            # choose one
            cos_list = []
            scale_list = []
            bias_list = []
            
            for chosen_i in ind:
                chosen_weight = weight[chosen_i]
                chosen_norm = np.linalg.norm(chosen_weight, ord = 2)
                chosen_cos = current_cos[chosen_i]
                gamma_2 = bn_weight[chosen_i]
                beta_2 = bn_bias[chosen_i]
                mu_2 = bn_mean[chosen_i]
                sigma_2 = bn_var[chosen_i]
                
                # compute cosine sim
                cos_list.append(chosen_cos)
                
                # compute s
                s = current_norm/chosen_norm
                
                # compute scale term
                scale_term_inference = s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2)
                scale_list.append(scale_term_inference)
                
                # compute bias term
                bias_term_inference = abs((gamma_2/sigma_2) * (s * (-(sigma_1*beta_1/gamma_1) + mu_1) - mu_2) + beta_2)

                bias_term_inference = bias_term_inference/scale_term_inference

                bias_list.append(bias_term_inference)

            assert(len(cos_list) == len(ind))
            assert(len(scale_list) == len(ind))
            assert(len(bias_list) == len(ind))
            

            # merge cosine distance and bias distance
            bias_list = (bias_list - np.min(bias_list)) / (np.max(bias_list)-np.min(bias_list))

            score_list = lam * np.array(cos_list) + (1-lam) * np.array(bias_list)


            # find index and scale with minimum distance
            min_ind = np.argmin(score_list)

            min_scale = scale_list[min_ind]
            min_cosine_sim = 1-cos_list[min_ind]

            # check threshold - second
            if threshold and min_cosine_sim < threshold:
                continue
            
            scaling_mat[i, min_ind] = min_scale

    return scaling_mat


class Decompose:
    def __init__(self, arch, param_dict, criterion, threshold, lamda, model_type, cfg, cuda):
        
        self.param_dict = param_dict
        self.arch = arch
        self.criterion = criterion
        self.threshold = threshold
        self.lamda = lamda
        self.model_type = model_type
        self.cfg = cfg
        self.cuda = cuda
        self.output_channel_index = {}
        self.decompose_weight = []

    def get_output_channel_index(self, value, layer_id):

        output_channel_index = []

        if len(value.size()) :

            weight_vec = value.view(value.size()[0], -1)
            weight_vec = weight_vec.cuda()

            # l1-norm
            if self.criterion == 'l1-norm':
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())
            
            # l2-norm
            elif self.criterion == 'l2-norm':
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:self.cfg[layer_id]]
                output_channel_index = sorted(arg_max_rev.tolist())

            # l2-GM
            elif self.criterion == 'l2-GM':
                weight_vec = weight_vec.cpu().detach().numpy()
                matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(np.abs(matrix), axis=0)

                output_channel_index = np.argpartition(similar_sum, -self.cfg[layer_id])[-self.cfg[layer_id]:]


        return output_channel_index




    def get_decompose_weight(self):

        # scale matrix
        z = None

        # copy original weight
        self.decompose_weight = list(self.param_dict.values())

        # cfg index
        layer_id = -1

        for index, layer in enumerate(self.param_dict):

            original = self.param_dict[layer]

            # VGG
            if self.arch == 'VGG':

                # feature
                if 'feature' in layer : 

                    # conv
                    if len(self.param_dict[layer].shape) == 4:

                        layer_id += 1

                        # get index
                        self.output_channel_index[index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        # Merge scale matrix 
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o


                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())

                        bn_weight = bn[index+1].cpu().detach().numpy()
                        bn_bias = bn[index+2].cpu().detach().numpy()
                        bn_mean = bn[index+3].cpu().detach().numpy()
                        bn_var = bn[index+4].cpu().detach().numpy()

                        x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(), np.array(self.output_channel_index[index]), self.threshold, 
                                                                                bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.model_type)

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[self.output_channel_index[index],:,:,:]

                        # update next input channel
                        input_channel_index = self.output_channel_index[index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned


                    # batchNorm
                    elif len(self.param_dict[layer].shape):
                        
                        # pruned
                        pruned = self.param_dict[layer][input_channel_index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                # first classifier
                else:
                    pruned = torch.zeros(original.shape[0],z.shape[0])

                    if self.cuda:
                        pruned = pruned.cuda()

                    for i, f in enumerate(original):
                        o_old = f.view(z.shape[1],-1)
                        o = torch.mm(z,o_old).view(-1)
                        pruned[i,:] = o            

                    self.decompose_weight[index] = pruned

                    break
            
            # ResNet
            elif self.arch == 'ResNet':

                # block
                if 'layer' in layer : 

                    # last layer each block
                    if '0.conv1.weight' in layer : 
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer :

                        # get index
                        self.output_channel_index[index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())

                        bn_weight = bn[index+1].cpu().detach().numpy()
                        bn_bias = bn[index+2].cpu().detach().numpy()
                        bn_mean = bn[index+3].cpu().detach().numpy()
                        bn_var = bn[index+4].cpu().detach().numpy()

                        x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(), np.array(self.output_channel_index[index]), self.threshold, 
                                                                                bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.model_type)

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[self.output_channel_index[index],:,:,:]

                        # update next input channel
                        input_channel_index = self.output_channel_index[index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned


                    # batchNorm
                    elif 'bn1' in layer :

                        if len(self.param_dict[layer].shape):

                            # pruned
                            pruned = self.param_dict[layer][input_channel_index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned
                    
                    # Merge scale matrix 
                    elif 'conv2' in layer :

                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        # update decompose weight
                        self.decompose_weight[index] = scaled
                

            # WideResNet
            elif self.arch == 'WideResNet':

                # block
                if 'block' in layer : 

                    # last layer each block
                    if '0.conv1.weight' in layer : 
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer :

                        # get index
                        self.output_channel_index[index] = self.get_output_channel_index(self.param_dict[layer], layer_id)

                        # make scale matrix with batchNorm
                        bn = list(self.param_dict.values())

                        bn_weight = bn[index+1].cpu().detach().numpy()
                        bn_bias = bn[index+2].cpu().detach().numpy()
                        bn_mean = bn[index+3].cpu().detach().numpy()
                        bn_var = bn[index+4].cpu().detach().numpy()

                        x = create_scaling_mat_conv_thres_bn(self.param_dict[layer].cpu().detach().numpy(), np.array(self.output_channel_index[index]), self.threshold, 
                                                                                bn_weight, bn_bias, bn_mean, bn_var, self.lamda, self.model_type)

                        z = torch.from_numpy(x).type(dtype=torch.float)

                        if self.cuda:
                            z = z.cuda()
                        
                        z = z.t()

                        # pruned
                        pruned = original[self.output_channel_index[index],:,:,:]

                        # update next input channel
                        input_channel_index = self.output_channel_index[index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned


                    # BatchNorm
                    elif 'bn2' in layer :

                        if len(self.param_dict[layer].shape):

                            # pruned
                            pruned = self.param_dict[layer][input_channel_index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned
                    

                    # Merge scale matrix
                    elif 'conv2' in layer :

                        # scale 
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        # update decompose weight
                        self.decompose_weight[index] = scaled

            # LeNet_300_100
            elif self.arch == 'LeNet_300_100':

                # ip
                if layer in ['ip1.weight','ip2.weight'] : 

                    # Merge scale matrix
                    if z != None:
                        original = torch.mm(original,z)

                    layer_id += 1


                    # concatenate weight and bias
                    if layer in 'ip1.weight' :
                        weight = self.param_dict['ip1.weight'].cpu().detach().numpy()
                        bias = self.param_dict['ip1.bias'].cpu().detach().numpy()

                    elif layer in 'ip2.weight' :
                        weight = self.param_dict['ip2.weight'].cpu().detach().numpy()
                        bias = self.param_dict['ip2.bias'].cpu().detach().numpy()

                    bias_reshaped = bias.reshape(bias.shape[0],-1)
                    concat_weight = np.concatenate([weight, bias_reshaped], axis = 1)

                    
                    # get index
                    self.output_channel_index[index] = self.get_output_channel_index(torch.from_numpy(concat_weight), layer_id)

                    # make scale matrix with bias
                    x = create_scaling_mat_ip_thres_bias(concat_weight, np.array(self.output_channel_index[index]), self.threshold, self.model_type)
                    z = torch.from_numpy(x).type(dtype=torch.float)
                    
                    if self.cuda:
                        z = z.cuda()
                    

                    # pruned
                    pruned = original[self.output_channel_index[index],:]

                    # update next input channel
                    input_channel_index = self.output_channel_index[index]

                    # update decompose weight
                    self.decompose_weight[index] = pruned

                elif layer in 'ip3.weight':

                    original = torch.mm(original,z)

                    # update decompose weight
                    self.decompose_weight[index] = original

                # update bias
                elif layer in ['ip1.bias','ip2.bias']:
                    self.decompose_weight[index] = original[input_channel_index]
                
                else :
                    pass                    
                    

    def main(self):

        if self.cuda == False:
            for layer in self.param_dict:
                self.param_dict[layer] = self.param_dict[layer].cpu()

        self.get_decompose_weight()

        return self.decompose_weight