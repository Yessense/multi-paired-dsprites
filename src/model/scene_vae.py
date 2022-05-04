import random
from argparse import ArgumentParser
from typing import Tuple, Optional, List

import pytorch_lightning as pl
import torch.optim

import wandb

from src.model.mlp import MLP
from src.utils import iou_pytorch  # type: ignore
from src.model.decoder import Decoder  # type: ignore
from src.model.encoder import Encoder  # type: ignore
from vsa import ItemMemory  # type: ignore

torch.set_printoptions(sci_mode=False)


class MultiPairedDspritesVAE(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("MultiDisDspritesVAE")
        parser.add_argument("--lr", type=float, default=0.0005,
                            help="model's learning rate")
        parser.add_argument("--image_size", type=int, default=(1, 64, 64), nargs=3,
                            help="size of input image")
        parser.add_argument("--latent_dim", type=int, default=1024,
                            help="dimension of the latent feature representation")
        parser.add_argument("--n_features", type=int, default=5,
                            help="number of different features")
        parser.add_argument("--obj_placeholders", type=bool, default=True,
                            help="object level placeholders")
        parser.add_argument("--feature_placeholders", type=bool, default=True,
                            help="feature level placeholders")
        parser.add_argument("--encoder_state_dict", type=str,
                            default='model/saved_states/encoder_state_dict.pt')
        return parent_parser

    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 1024,
                 lr: float = 0.001,
                 n_features: int = 5,
                 feature_placeholders: bool = False,
                 obj_placeholders: bool = False,
                 feature_names: Optional[List] = None,
                 obj_names: Optional[List] = None,
                 encoder_state_dict: Optional[str] = '',
                 **kwargs):
        super().__init__()

        # Model parameters
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.lr = lr

        self.dataset_mode = kwargs['dataset_mode']

        # Load Encoder from state dict
        self.encoder = Encoder(latent_dim=self.latent_dim, image_size=self.image_size, n_features=self.n_features)

        if encoder_state_dict is not None and len(encoder_state_dict):
            state_dict = torch.load(encoder_state_dict)
            self.encoder.load_state_dict(state_dict)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        # Decoder
        self.decoder = Decoder(latent_dim=self.latent_dim, image_size=self.image_size, n_features=self.n_features)

        # MLP if needed
        # if self.dataset_mode == 'two objects':
        self.mlp = MLP(latent_dim=self.latent_dim)

        # Feature placeholders
        self.feature_placeholders = feature_placeholders
        # placeholder vectors -> (1, 5, 1024) for multiplication on object features
        # ready to .expand()
        if self.feature_placeholders:
            if feature_names is None:
                self.feature_names = ['shape', 'size', 'rotation', 'posx', 'posy']
            else:
                self.feature_names = feature_names

            features_im: ItemMemory = ItemMemory(name="Features", dimension=self.latent_dim,
                                                 init_vectors=self.feature_names)
            self.hd_feature_placeholders = torch.Tensor(features_im.memory).float().to(self.device)
            self.hd_feature_placeholders = self.hd_feature_placeholders.unsqueeze(0)

        # Object placeholders
        self.obj_placeholders = obj_placeholders
        # placeholder vector -> (2, 1024) = [1024, 1024] for multiplication on objects
        # ready to .expand()
        if self.obj_placeholders:
            if obj_names is None:
                self.obj_names = ['obj1', 'obj2']
            else:
                self.obj_names = obj_names

            objs_im: ItemMemory = ItemMemory(name="Objects", dimension=self.latent_dim,
                                             init_vectors=self.obj_names)
            self.hd_obj_placeholders = [torch.Tensor(objs_im.get_vector(name).vector).float().to(self.device) for name
                                        in
                                        self.obj_names]

        self.save_hyperparameters()

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode_image(self, image, placeholders=False):
        """Multiply img features on feature placeholders"""
        mu, log_var = self.encoder(image)

        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        # z -> (-1, 5,  1024)
        z = z.view(-1, 5, self.latent_dim)

        if placeholders:
            # mask -> (-1, 5, 1024)
            mask = self.hd_feature_placeholders.expand(z.size()).to(self.device)
            z = z * mask

        return z

    def encode_scene(self, z1, z2):
        """Make scene from sum of features vectors"""
        if self.obj_placeholders:
            batch_size = z1.shape[0]
            masks = [mask.repeat(batch_size, 1).to(self.device) for mask in self.hd_obj_placeholders]
            z1 *= masks[0]
            z2 *= masks[1]

        scene = z1 + z2

        return scene

    def training_step(self, batch):
        """Function exchanges objects from scene1 to scene2"""
        if self.dataset_mode == 'exchange':
            img, img2, donor, pair_img, scene, exchange_labels = batch
            # scene, img, pair_img = batch

            # Encode features
            latent_img = self.encode_image(img, self.feature_placeholders)
            latent_donor = self.encode_image(donor, self.feature_placeholders)
            latent_img2 = self.encode_image(img2, self.feature_placeholders)

            # Expand exchange labels (-1, 5, 1) -> (-1, 5, 1024)
            exchange_labels = exchange_labels.expand(latent_img.size())

            # Exchange one feature between latent img and latent donor
            latent_pair_img = torch.where(exchange_labels, latent_img, latent_donor)

            # Sum latent features into object
            latent_img2 = torch.sum(latent_img2, dim=1)
            latent_pair_img = torch.sum(latent_pair_img, dim=1)

            # encode scene
            batch_size = latent_img.shape[0]
            masks = [mask.repeat(batch_size, 1).to(self.device) for mask in self.hd_obj_placeholders]

            # Multiply on placeholder vector
            if self.global_step % 2 == 0:
                latent_img *= masks[0]
                latent_pair_img *= masks[1]

                scene_latent = latent_img + latent_pair_img

                reconstructed_img = self.decoder(self.mlp(torch.cat([scene_latent, masks[0]], dim=1)))
                reconstructed_pair_img = self.decoder(self.mlp(torch.cat([scene_latent, masks[1]], dim=1)))
            else:
                latent_img *= masks[1]
                latent_pair_img *= masks[0]

                scene_latent = latent_img + latent_pair_img

                reconstructed_img = self.decoder(self.mlp(torch.cat([scene_latent, masks[1]], dim=1)))
                reconstructed_pair_img = self.decoder(self.mlp(torch.cat([scene_latent, masks[0]], dim=1)))

            img_loss = self.loss_f(reconstructed_img, img)
            pair_loss = self.loss_f(reconstructed_pair_img, pair_img)
            loss = img_loss + pair_loss
            iou1 = iou_pytorch(reconstructed_img, img)
            iou2 = iou_pytorch(reconstructed_pair_img, pair_img)
            iou = iou1 + iou2
            iou /= 2

            # log training process
            self.log("BCE reconstruct", loss)
            self.log("Image BCE loss", img_loss)
            self.log("Pair image BCE loss", pair_loss)

            self.log("IOU Image", iou1)
            self.log("IOU Pair image", iou2)
            self.log("IOU", iou, prog_bar=True)

            # log images
            if self.global_step % 499 == 0:
                self.logger.experiment.log({
                    "reconstruct/examples": [
                        wandb.Image(img[0], caption='Image 1'),
                        wandb.Image(img2[0], caption='Image 2'),
                        wandb.Image(donor[0], caption='Donor'),
                        wandb.Image(pair_img[0], caption='Pair image'),
                        wandb.Image(scene[0], caption='Scene'),
                        wandb.Image(reconstructed_img[0], caption='Reconstructed image'),
                        wandb.Image(reconstructed_pair_img[0], caption='Reconstructed pair image'),
                    ]})
            return loss

        # Just two objects that wont't instersect
        if self.dataset_mode == 'two objects':
            scene, img, pair_img = batch

            # encode features
            latent_img = self.encode_image(img, self.feature_placeholders)
            latent_pair_img = self.encode_image(pair_img, self.feature_placeholders)

            # Sum latent features into object
            latent_img = torch.sum(latent_img, dim=1)
            latent_pair_img = torch.sum(latent_pair_img, dim=1)

            # encode scene
            batch_size = latent_img.shape[0]
            masks = [mask.repeat(batch_size, 1).to(self.device) for mask in self.hd_obj_placeholders]
            latent_img *= masks[0]
            latent_pair_img *= masks[1]

            scene_latent = latent_img + latent_pair_img

            reconstructed_img = self.decoder(self.mlp(torch.cat([scene_latent, masks[0]], dim=1)))
            reconstructed_pair_img = self.decoder(self.mlp(torch.cat([scene_latent, masks[1]], dim=1)))

            loss = self.loss_f(reconstructed_img, img) + self.loss_f(reconstructed_pair_img, pair_img)
            iou = iou_pytorch(reconstructed_img, img) + iou_pytorch(reconstructed_pair_img, pair_img)
            iou /= 2

            # log training process
            self.log("BCE reconstruct", loss)
            self.log("IOU", iou, prog_bar=True)

            # log images
            if self.global_step % 499 == 0:
                self.logger.experiment.log({
                    "reconstruct/examples": [
                        wandb.Image(img[0], caption='Image'),
                        wandb.Image(pair_img[0], caption='Pair image'),
                        wandb.Image(reconstructed_img[0], caption='Reconstructed Image'),
                        wandb.Image(reconstructed_pair_img[0], caption='Reconstructed Pair image'),
                    ]})

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_f(self, reconstruct, scene):
        loss_func = torch.nn.BCELoss(reduction='sum')
        loss = loss_func(reconstruct, scene)
        return loss
