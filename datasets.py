import h5py
import torch
import torchvision
import collections

import numpy as np

def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict

class CSWMDataset(torch.utils.data.Dataset):

    def __init__(self, fname):

        self.data = load_list_dict_h5py(fname)
        self.obs = []
        self.next_obs = []
        self.action = []
        for d in self.data:
            self.obs.extend(d['obs'])
            self.next_obs.extend(d['next_obs'])
            self.action.extend(d['action'])

        self.data_length = len(self.obs)

    def __getitem__(self, idx):
        negative_idx = idx
        while negative_idx == idx:
            negative_idx = np.random.choice(self.data_length)

        negative_obs = self.obs[negative_idx]
        positive_obs = self.obs[idx]
        next_obs = self.next_obs[idx]
        action = self.action[idx]

        negative_obs = torch.as_tensor(negative_obs, dtype=torch.float32)
        positive_obs = torch.as_tensor(positive_obs, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.long)

        return negative_obs, positive_obs, next_obs, action

    def __len__(self):
        return len(self.obs)
        

if __name__ == '__main__':
    
    dataset = CSWMDataset('shapes_train.h5')

    import matplotlib
    matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt

    negative_obs, positive_obs, next_obs, action = dataset[0]

    negative_obs = negative_obs.permute([1, 2, 0]).detach().cpu().numpy()
    positive_obs = positive_obs.permute([1, 2, 0]).detach().cpu().numpy()
    next_obs = next_obs.permute([1, 2, 0]).detach().cpu().numpy()
    
    print(action)
    print('------')

    plt.subplot(1,3,1)
    plt.imshow(negative_obs)
    plt.subplot(1,3,2)
    plt.imshow(positive_obs)
    plt.subplot(1,3,3)
    plt.imshow(next_obs)
    plt.show()