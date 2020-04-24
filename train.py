import torch
import torchvision
import datasets
import module
import loss

import torch.nn.functional as F

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.CSWMDataset('shapes_train.h5')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=4,
        batch_size=32)
    
    model = module.Model()
    model = model.to(device)

    latent_parameter = list(model.object_extractor.parameters()) + \
                       list(model.object_encoder.parameters()) + \
                       list(model.transition_model.parameters())

    reconstruct_parameter = list(model.object_decoder.parameters())

    # latent_optimizer = torch.optim.Adam(
    #     params=latent_parameter, lr=1e-4)

    # reconstruct_optimizer = torch.optim.Adam(
    #     params=reconstruct_parameter, lr=1e-4)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=1e-4)

    for step, (negative_obs, positive_obs, next_obs, action) in enumerate(dataloader):

        negative_obs = negative_obs.to(device)
        positive_obs = positive_obs.to(device)
        next_obs = next_obs.to(device)
        action = action.to(device)

        # latent_optimizer.zero_grad()
        # reconstruct_optimizer.zero_grad()
        optimizer.zero_grad()

        positive_obs_z, positive_node, positive_next_obs_z = model(positive_obs, next_obs, action)
        negative_obs_z, negative_node, negative_next_obs_z = model(negative_obs, next_obs, action)

        predict_obs = torch.sigmoid(model.reconstruct(positive_obs_z + positive_node))

        latent_loss = loss.contrastive_loss(
            negative_obs_z, positive_obs_z,
            negative_node, positive_node,
            negative_next_obs_z, positive_next_obs_z)

        reconstruction_loss = F.binary_cross_entropy(
            predict_obs, next_obs, reduction='sum') / positive_node.shape[0]

        total_loss = reconstruction_loss + latent_loss

        total_loss.backward()
        optimizer.step()
        # latent_optimizer.step()
        # reconstruct_optimizer.step()

        print('-----------------')
        print(f'step                : {step}')
        print(f'latent loss         : {latent_loss.item()}')
        print(f'reconstruction loss : {reconstruction_loss.item()}')
        print('-----------------')

        if step % 300 == 0:

            torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    train()