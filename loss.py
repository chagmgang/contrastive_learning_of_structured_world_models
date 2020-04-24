import torch
import torchvision

def contrastive_loss(negative_obs_z, positive_obs_z,
                     negative_node, positive_node,
                     negative_next_obs_z, positive_next_obs_z):

    batch_size, object_num, feature = negative_next_obs_z.shape

    H = []
    for b in range(batch_size):
        HH = []
        for num in range(object_num):
            z_t = positive_obs_z[b, num]
            T = positive_node[b, num]
            z_t_1 = positive_next_obs_z[b, num]
            dist = torch.dist(
                z_t + T,
                z_t_1,
                p=2)
            HH.append(dist)
        HH = torch.stack(HH)
        HH = torch.mean(HH)
        H.append(HH)
    H = torch.stack(H)
    
    H_hat = []
    for b in range(batch_size):
        HH_hat = []
        for num in range(object_num):
            hat_z_t = negative_obs_z[b, num]
            z_t_1 = negative_next_obs_z[b, num]
            dist = torch.dist(
                hat_z_t,
                z_t_1,
                p=2)
            HH_hat.append(dist)
        HH_hat = torch.stack(HH_hat)
        HH_hat = torch.mean(HH_hat)
        H_hat.append(HH_hat)
    H_hat = torch.stack(H_hat)
    H_hat = 1 - H_hat
    
    H_hat_zero = torch.zeros_like(H_hat)
    loss = H + torch.where(H_hat < 0, H_hat_zero, H_hat)
    return torch.mean(loss)