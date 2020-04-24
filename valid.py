import torch
import torchvision
import datasets
import module
import loss

import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.CSWMDataset('shapes_train.h5')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=4,
        batch_size=32,
        shuffle=True)
    
    model = module.Model()
    model = model.to(device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    negative_obs, positive_obs, next_obs, action = next(iter(dataloader))

    negative_obs = negative_obs.to(device)
    positive_obs = positive_obs.to(device)
    next_obs = next_obs.to(device)
    action = action.to(device)

    positive_obs_z, positive_node, positive_next_obs_z = model(positive_obs, next_obs, action)
    predict_next_obs = torch.sigmoid(model.reconstruct(positive_obs_z + positive_node))

    sample_idx = np.random.choice(32)
    print(f'sampling idx : {sample_idx}')
    sample_idx = positive_obs_z[sample_idx] + positive_node[sample_idx]

    dist = [torch.dist(sample_idx, positive_next_obs_z[i], p=2) for i in range(32)]
    dist = torch.stack(dist)
    sort = torch.argsort(dist)
    print(sort)

    predict_next_obs = predict_next_obs[0].permute([1, 2, 0]).detach().cpu().numpy()
    positive_obs = positive_obs[0].permute([1, 2, 0]).detach().cpu().numpy()
    next_obs = next_obs[0].permute([1, 2, 0]).detach().cpu().numpy()

    plt.subplot(1,3,1)
    plt.imshow(predict_next_obs)
    plt.subplot(1,3,2)
    plt.imshow(next_obs)
    plt.subplot(1,3,3)
    plt.imshow(positive_obs)
    plt.show()
        

if __name__ == '__main__':
    train()