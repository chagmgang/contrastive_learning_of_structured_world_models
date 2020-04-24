import torch
import torchvision

import torch.nn as nn
import numpy as np

class ObjectExtractor(nn.Module):

    def __init__(self):
        super(ObjectExtractor, self).__init__()

        self.c1 = nn.Conv2d(
            in_channels=3,
            out_channels=256,
            kernel_size=(10, 10),
            stride=10)

        self.c2 = nn.Conv2d(
            in_channels=256,
            out_channels=5,
            kernel_size=(1, 1),
            stride=1)

        self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        return x

class ObjectEncoder(nn.Module):

    def __init__(self):
        super(ObjectEncoder, self).__init__()

        self.l1 = nn.Linear(5 * 5, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)

        self.layer_norm = nn.LayerNorm(256)

        self.relu = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, num_objects, height, width = x.shape
        x = torch.reshape(x, (batch_size, num_objects, height * width))
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.layer_norm(x)
        x = self.l3(x)
        return x

class ObjectDecoder(nn.Module):

    def __init__(self):
        super(ObjectDecoder, self).__init__()
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 25)
        self.ln = nn.LayerNorm(256)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=5,
            out_channels=256,
            kernel_size=1, stride=1)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=3,
            kernel_size=10, stride=10)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, _ = x.shape
        x = self.l1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.l2(x)
        x = torch.reshape(x, (batch_size, channel, 5, 5))
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x

class TransitionModel(nn.Module):

    def __init__(self):
        super(TransitionModel, self).__init__()

        self.action_embedding = nn.Embedding(
            num_embeddings=20, embedding_dim=256)

        self.edge_mlp = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 256))

        self.node_mlp = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 256))

    def get_sigma(self, x):
        sum_edge_j = []
        for i in range(5):
            sum_edge = 0
            for j in range(5):
                if not i == j:
                    source = x[:, i]
                    target = x[:, j]
                    cat = torch.cat([source, target], axis=1)
                    edge = self.edge_mlp(cat)
                    sum_edge += edge
            sum_edge_j.append(sum_edge)
        sum_edge_j = torch.stack(sum_edge_j)
        return sum_edge_j.permute([1, 0, 2])
    
    def forward(self, x, action):
        batch_size, kernel_size, _ = x.shape
        action_embedding = self.action_embedding(action)
        action_embedding = torch.reshape(action_embedding, (batch_size, 1, -1))
        action_embedding = action_embedding.repeat(1, kernel_size, 1)
        edge_sum = self.get_sigma(x)

        cat = torch.cat([x, action_embedding, edge_sum], axis=2)
        node = self.node_mlp(cat)
        return node

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.object_extractor = ObjectExtractor()
        self.object_encoder = ObjectEncoder()
        self.transition_model = TransitionModel()
        self.object_decoder = ObjectDecoder()

    def forward(self, obs, next_obs, action):
        obs_m = self.object_extractor(obs)
        next_obs_m = self.object_extractor(next_obs)

        obs_z = self.object_encoder(obs_m)
        next_obs_z = self.object_encoder(next_obs_m)

        node = self.transition_model(obs_z, action)

        return obs_z, node, next_obs_z

    def reconstruct(self, latent_vector):
        reconstructed_image = self.object_decoder(latent_vector)
        return reconstructed_image

if __name__ == '__main__':
    import datasets
        
    dataset = datasets.CSWMDataset('shapes_train.h5')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=4,
        batch_size=32,
        shuffle=True)

    model = Model()

    negative_obs, positive_obs, next_obs, action = next(iter(dataloader))
    node, next_obs_z = model(negative_obs, next_obs, action)
    print(node.shape, next_obs_z.shape)
    # model(positive_obs, next_obs, action)