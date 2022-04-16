from argparse import ArgumentParser
from typing import List, Iterable, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.dataset.dataset import MultiDisDsprites
from src.model.scene_vae import MultiPairedDspritesVAE

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--checkpoint_path", type=str,
                            default='/home/yessense/PycharmProjects/Multi-paired-dsprites/src/model/checkpoint/epoch=98-step=96722.ckpt')
program_parser.add_argument("--batch_size", type=int, default=5)

# parse input
args = parser.parse_args()


def load_model(checkpoint_path: str) -> MultiPairedDspritesVAE:
    """Load MultiDisDspritesVae model from checkpoint"""
    ckpt = torch.load(checkpoint_path)

    hyperparams = ckpt['hyper_parameters']
    hyperparams['encoder_state_dict'] = '/home/yessense/PycharmProjects/Multi-paired-dsprites/src/model/saved_states/encoder_state_dict.pt'
    state_dict = ckpt['state_dict']

    model = MultiPairedDspritesVAE(**hyperparams)
    model.load_state_dict(state_dict)
    return model


class Experiment:
    def __init__(self, checkpoint_path: str,
                 dataset_path: str = '/home/yessense/PycharmProjects/Multi-paired-dsprites/src/dataset/data/dsprites.npz'):

        # Load model
        self.device = torch.device('cuda:0')
        self.model = load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Load dataset
        self.dataset = MultiDisDsprites(dataset_path)

    def reconstruct_scene(self):
        self.dataset

    def exchange_feature(self, feat1, feat2, n_feature: int):
        """Exchange N-th feature between feat1 and feat2"""

        exchange_label = torch.ones(1, 5, 1024)
        exchange_label[:, n_feature, :] = 0
        exchange_label = exchange_label.to(self.model.device).bool()

        out = torch.where(exchange_label, feat1, feat2)
        return out

    def get_decoded_scene(self, latent1, latent2):
        """Build scene from 2 vectors of features"""
        scene = self.model.encode_scene(z1=latent1, z2=latent2)

        z = scene.sum(dim=1)

        reconstructed = self.model.decoder(z)
        return reconstructed

    def plot_list(self, suptitle: str, titles: Sequence[str], images: List):
        """Plot a sequence of images in a row"""

        fig, ax = plt.subplots(1, len(images))
        plt.figure(figsize=(20, 8))

        if len(images) == 1:
            ax.imshow(images[1][0].detach().cpu().numpy().squeeze(0), cmap='gray')
            ax.set_axis_off()
        else:
            for i in range(len(images)):
                ax[i].imshow(images[i][0].detach().cpu().numpy().squeeze(0), cmap='gray')
                ax[i].set_axis_off()
                ax[i].set_title(titles[i])

        fig.suptitle(suptitle)
        plt.show()

    def reconstruct_features(self, placeholders=False):
        """Reconstruct each feature"""
        scene, image1, donor, image2 = self.dataset.inference_sample()

        image = torch.unsqueeze(image1, dim=0)
        image = image.to(self.device)

        z = self.model.encode_image(image, placeholders=placeholders)

        features_decoded = [self.model.decoder(feature.unsqueeze(0)) for feature in z[0, :, :]]
        labels = ('image', 'shape', 'scale', 'orientation', 'posX', 'posY')

        self.plot_list(suptitle=f'Reconstruction of features, placeholders = {placeholders}', titles=labels,
                       images=[image] + features_decoded)

    def exchange_features(self):
        """Change each feature between 2 images on scene"""
        scene, image1, donor, image2 = self.dataset.inference_sample()

        # make (1, 64, 64) -> (1, 1, 64, 64) batch like
        image1 = torch.unsqueeze(image1, dim=0)
        image2 = torch.unsqueeze(image2, dim=0)
        donor = torch.unsqueeze(donor, dim=0)
        scene = torch.unsqueeze(scene, dim=0)

        # self.plot_list(suptitle='Images in experiment',
        #                titles=['Image 1', 'Image 2', 'Donor', 'Scene'],
        #                images=[image1, image2, donor, scene])

        # move to gpu
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        donor = donor.to(self.device)

        # get latent representations
        latent_i1 = self.model.encode_image(image1, placeholders=self.model.hd_features)
        latent_d = self.model.encode_image(donor, placeholders=self.model.hd_features)
        latent_i2 = self.model.encode_image(image2, placeholders=self.model.hd_features)

        fig, ax = plt.subplots(5, 5)
        plt.figure(figsize=(20, 8))

        y_labels = ('shape', 'scale', 'orientation', 'posX', 'posY')
        x_labels = ('Decoded scene', 'Image 1', 'Image 2', 'Donor', 'Scene')

        for i in range(5):
            exchanged_latent = self.exchange_feature(latent_i1, latent_d, n_feature=i)
            exchanged_scene = self.get_decoded_scene(exchanged_latent, latent_i2)

            for j, img in enumerate([exchanged_scene, image1, image2, donor, scene]):
                ax[i, j].imshow(img[0].detach().cpu().numpy().squeeze(0), cmap='gray')
                if j == 0:
                    ax[i, j].set_ylabel(y_labels[i])
                if i == 4:
                    ax[i, j].set_xlabel(x_labels[j])
                if j != 0 and i != 4:
                    ax[i, j].set_axis_off()
        fig.suptitle('Exchanges of N-th feature between `image1` and `donor`')
        plt.show()


if __name__ == '__main__':
    experiment = Experiment(args.checkpoint_path)
    experiment.exchange_features()
    # experiment.reconstruct_features(placeholders=False)
    # experiment.reconstruct_features(placeholders=True)

    print("Done")
