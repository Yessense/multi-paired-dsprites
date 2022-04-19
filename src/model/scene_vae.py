import random
from argparse import ArgumentParser
from typing import Tuple, Optional, List

import pytorch_lightning as pl
import torch.optim

import wandb

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

        # Load Encoder from state dict
        self.encoder = Encoder(latent_dim=self.latent_dim, image_size=self.image_size, n_features=self.n_features)

        if encoder_state_dict is None or not len(encoder_state_dict):
            raise NameError

        state_dict = torch.load(encoder_state_dict)
        self.encoder.load_state_dict(state_dict)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # Decoder
        self.decoder = Decoder(latent_dim=self.latent_dim, image_size=self.image_size, n_features=self.n_features)

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
        img, img2, donor, pair_img, scene, exchange_labels = batch
        # scene, img, pair_img = batch

        # Encode features
        latent_img = self.encode_image(img, self.feature_placeholders)
        latent_donor = self.encode_image(donor, self.feature_placeholders)
        latent_img2 = self.encode_image(img2, self.feature_placeholders)

        exchange_labels = exchange_labels.expand(latent_img.size())

        latent_pair_img = torch.where(exchange_labels, latent_img, latent_donor)

        latent_img2 = torch.sum(latent_img2, dim=1)
        latent_pair_img = torch.sum(latent_pair_img, dim=1)

        if self.global_step % 2 == 0:
            scene_latent = self.encode_scene(latent_img2, latent_pair_img)
        else:
            scene_latent = self.encode_scene(latent_pair_img, latent_img2)

        reconstruct = self.decoder(scene_latent)

        loss = self.loss_f(reconstruct, scene)
        iou = iou_pytorch(reconstruct, scene)

        # log training process
        self.log("BCE reconstruct", loss)
        self.log("IOU", iou, prog_bar=True)

        if self.global_step % 499 == 0:
            self.logger.experiment.log({
                "reconstruct/examples": [
                    wandb.Image(scene[0], caption='Scene 1'),
                    wandb.Image(reconstruct[0], caption='Recon 1'),
                ]})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_f(self, reconstruct, scene):
        loss_func = torch.nn.BCELoss(reduction='sum')
        loss = loss_func(reconstruct, scene)
        return loss
