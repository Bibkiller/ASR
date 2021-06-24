import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq

# Reconstruction
def reconst(vec_s,query_gro,nor_sum,base_dim,num_class):
    '''
    vec_s: support vector
    query_gro: grouped query features
    nor_sum: The sum of normed weights
    base_dim: dim of basis vectors
    num_class: number of base classes
    '''
    vec_s = vec_s/(nor_sum+1e-7)
    for i in range(num_class):
        base = vec_s[:,base_dim*i:base_dim*(i+1),:,:]
        vec_reconst += base 
    for i in range(num_class): 
        qfeat_reconst+=query_gro[:,i*base_dim:(i+1)*base_dim,:,:] 
            

def projection(vec_r,feat_r):
    '''
    vec_r: reconstructed vector
    feat_r: reconstructed features
    '''
    eps=1e-7
    cos_map = torch.cosine_similarity(feat_r, vec_r, dim=1)
    feat_norm = torch.norm(feat_r,2,1)
    proj_norm = feat_norm*cos_map
    vec_norm = torch.norm(vec_r,2,1).unsqueeze(1)
    normed_vec = vec_r/(vec_norm+eps)
    vec_mat=normed_vec.expand(normed_vec.shape[0], normed_vec.shape[1], feat_r.shape[2],feat_r.shape[3])

    proj = torch.mul(vec_mat,proj_norm.unsqueeze(1))
    return proj, cos_map
    
def semantic_span(supp_gro, query_gro, cls_id, index, class_num, num_pairs=1):
    '''
    supp_gro: grouped support features
    query_gro: grouped query features
    '''
    query_pos = torch.cat([query_gro[0,cls_id[0]*index:cls_id[0]*index+index,:,:].unsqueeze(0),
                            query_gro[1,cls_id[1]*index:cls_id[1]*index+index,:,:].unsqueeze(0),query_gro[2,cls_id[2]*index:cls_id[2]*index+index,:,:].unsqueeze(0),
                            query_gro[3,cls_id[3]*index:cls_id[3]*index+index,:,:].unsqueeze(0)],dim=0)
    supp_pos = torch.cat([supp_gro[0,cls_id[0]*index:cls_id[0]*index+index,:,:].unsqueeze(0),supp_gro[1,cls_id[1]*index:cls_id[1]*index+index,:,:].unsqueeze(0),
                              supp_gro[2,cls_id[2]*index:cls_id[2]*index+index,:,:].unsqueeze(0),supp_gro[3,cls_id[3]*index:cls_id[3]*index+index,:,:].unsqueeze(0)],dim=0)
    for i in range(query_gro.shape[0]):
        index_list = list()
        for j in range(index):
            index_list.append(cls_id[i]*index+j)
        if i==0:
            query_negs = torch.from_numpy(np.delete(query_gro[i,:,:,:].detach().cpu().numpy(), index_list, axis = 0)).unsqueeze(0).cuda()
        else:
            query_negs = torch.cat([query_negs,torch.from_numpy(np.delete(query_gro[i,:,:,:].detach().cpu().numpy(), index_list, axis = 0)).unsqueeze(0).cuda()],dim=0)
    q_negvec_list = list()
    for i in range(class_num-1):
        q_negvec = F.avg_pool2d(input=query_negs[:,index*i:index*(i+1),:,:],kernel_size=supp_gro.shape[-2:])
        q_negvec_list.append(q_negvec)
            
    supp_pos_vec = F.avg_pool2d(input=supp_pos,kernel_size=supp_gro.shape[-2:])
    query_pos_vec = F.avg_pool2d(input=query_pos,kernel_size=query_gro.shape[-2:])    
    cosine_pos = torch.abs(torch.cosine_similarity(supp_pos_vec.squeeze(2).squeeze(2), query_pos_vec.squeeze(2).squeeze(2), dim=1))
    cosine_negs = torch.zeros(4).cuda()
    for idx,qneg in enumerate(q_negvec_list):
        cosine_negs+=torch.abs(torch.cosine_similarity(supp_pos_vec.squeeze(2).squeeze(2), qneg.squeeze(2).squeeze(2), dim=1))
        if idx>num_pairs:
            break
    return [cosine_pos,cosine_negs]

def generate_vector(self, support_feature, support_mask):
    support_mask = F.interpolate(support_mask, support_feature.shape[-2:], mode='bilinear', align_corners=True)
    h, w = support_feature.shape[-2:][0], support_feature.shape[-2:][1]

    area = F.avg_pool2d(support_mask, support_feature.shape[-2:]) * h * w + 0.0005
    z = support_mask * support_feature
    vec_pos = F.avg_pool2d(input=z,
                               kernel_size=support_feature.shape[-2:]) * h * w / area
        
    return vec_pos,z,h * w / area
