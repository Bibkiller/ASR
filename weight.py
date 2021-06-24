import torch


def vec_weight(class_num,map_num,vec_support):
    class_map_list=list()
    for i in range(class_num):
#         class_weight_temp = torch.norm(vec_support[:,i*map_num:(i*map_num+map_num),:,:],p=1,dim=1) It seems that l1 norm works better sometimes.
        class_weight_temp = torch.norm(vec_support[:,i*map_num:(i*map_num+map_num),:,:],p=2,dim=1)
        class_weight_list.append(class_weight_temp.unsqueeze(1))
    return class_map_list