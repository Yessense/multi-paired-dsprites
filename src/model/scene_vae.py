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
        parser.add_argument("--hd_objs", type=bool, default=True,
                            help="object level placeholders")
        parser.add_argument("--hd_features", type=bool, default=True,
                            help="feature level placeholders")
        parser.add_argument("--kld_coef", type=float, default=0.001,
                            help="kl loss part coefficient")
        parser.add_argument("--encoder_state_dict", type=str,
                            default='/home/yessense/PycharmProjects/Multi-paired-dsprites/src/model/saved_states'
                                    '/encoder_state_dict.pt')
        return parent_parser

    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 1024,
                 lr: float = 0.001,
                 n_features: int = 5,
                 hd_objs: bool = False,
                 hd_features: bool = False,
                 feature_names: Optional[List] = None,
                 obj_names: Optional[List] = None,
                 kld_coef: float = 1.0,
                 encoder_state_dict: Optional[str] = '',
                 **kwargs):
        super().__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.lr = lr
        self.kld_coef = kld_coef

        self.encoder = Encoder(latent_dim=self.latent_dim, image_size=self.image_size, n_features=self.n_features)
        if encoder_state_dict is not None and len(encoder_state_dict):
            state_dict = torch.load(encoder_state_dict)
            self.encoder.load_state_dict(state_dict)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.decoder = Decoder(latent_dim=self.latent_dim, image_size=self.image_size, n_features=self.n_features)

        self.hd_objs = hd_objs
        self.hd_features = hd_features

        # placeholder vectors -> (1, 5, 1024) for multiplication on object features
        # ready to .expand()
        if self.hd_features:
            if feature_names is None:
                self.feature_names = ['shape', 'size', 'rotation', 'posx', 'posy']
            else:
                self.feature_names = feature_names

            features_im: ItemMemory = ItemMemory(name="Features", dimension=self.latent_dim,
                                                 init_vectors=self.feature_names)
            self.feature_placeholders = torch.Tensor(features_im.memory).float().to(self.device)
            self.feature_placeholders = self.feature_placeholders.unsqueeze(0)

        # placeholder vector -> (2, 1024) = [1024, 1024] for multiplication on objects
        # ready to .expand()
        if self.hd_objs:
            if obj_names is None:
                self.obj_names = ['obj1', 'obj2']
            else:
                self.obj_names = obj_names

            objs_im: ItemMemory = ItemMemory(name="Objects", dimension=self.latent_dim,
                                             init_vectors=self.obj_names)
            self.obj_placeholders = [torch.Tensor(objs_im.get_vector(name).vector).float().to(self.device) for name in
                                     self.obj_names]

        self.save_hyperparameters()

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def encode_features(self, image, placeholders=False):
        """Multiply img features on feature placeholders"""
        mu, log_var = self.encoder(image)

        # z -> (-1, 5,  1024)
        z = self.reparameterize(mu, log_var)
        z = z.view(-1, 5, self.latent_dim)

        if placeholders:
            # mask -> (-1, 5, 1024)
            mask = self.feature_placeholders.expand(z.size()).to(self.device)
            z = z * mask

        if self.training:
            return mu, log_var, z
        else:
            return z

    def encode_scene(self, z1, z2):
        batch_size = z1.shape[0]
        masks = [mask.repeat(batch_size, 1).to(self.device) for mask in self.obj_placeholders]

        # 0 or 1
        choice = random.randint(0, 1)

        z1 *= masks[choice]
        z2 *= masks[not choice]

        scene = z1 + z2

        return scene

    def training_step(self, batch):
        """Function exchanges objects from scene1 to scene2"""
        scene1, scene2, fist_obj, pair_obj, second_obj, exchange_label = batch

        # Encode features
        mu1, log_var1, feat_1 = self.encode_features(fist_obj, self.hd_features)
        mu2, log_var2, feat_2 = self.encode_features(pair_obj, self.hd_features)
        mu3, log_var3, z3 = self.encode_features(second_obj, self.hd_features)

        # exchange labels -> (-1, 5, 1024)
        exchange_label = exchange_label.expand(feat_1.size())

        # z1 Восстанавливает 1 изображение
        z1 = torch.where(exchange_label, feat_1, feat_2)
        # z2 Восстанавливает 2 изображение изображение
        z2 = torch.where(exchange_label, feat_2, feat_1)

        # z1 -> first object -> (-1, 1024)
        z1 = torch.sum(z1, dim=1)
        # z2 -> pair object -> (-1, 1024)
        z2 = torch.sum(z2, dim=1)
        # z3 -> second object -> (-1, 1024)
        z3 = torch.sum(z3, dim=1)

        # multiply by object number placeholders
        scene1_latent = self.encode_scene(z1, z3)
        scene2_latent = self.encode_scene(z2, z3)

        r1 = self.decoder(scene1_latent)
        r2 = self.decoder(scene2_latent)

        total, l1, l2 = self.loss_f(r1, r2, scene1, scene2)
        iou1 = iou_pytorch(r1, scene1)
        iou2 = iou_pytorch(r2, scene2)
        iou = (iou1 + iou2) / 2

        # log training process
        self.log("Sum of losses", total, prog_bar=True)
        self.log("BCE reconstruct 1, img 1", l1, prog_bar=False)
        self.log("BCE reconstruct 2, img 2", l2, prog_bar=False)
        self.log("IOU mean ", iou, prog_bar=True)
        self.log("IOU reconstruct 1, img 1", iou1, prog_bar=False)
        self.log("IOU reconstruct 2, img 2", iou2, prog_bar=False)

        if self.global_step % 499 == 0:
            self.logger.experiment.log({
                "reconstruct/examples": [
                    wandb.Image(scene1[0], caption='Scene 1'),
                    wandb.Image(scene2[0], caption='Scene 2'),
                    wandb.Image(r1[0], caption='Recon 1'),
                    wandb.Image(r2[0], caption='Recon 2'),
                    wandb.Image(fist_obj[0], caption='Image 1'),
                    wandb.Image(pair_obj[0], caption='Pair to Image 1'),
                    wandb.Image(second_obj[0], caption='Image 2')
                ]})

        return total

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_f(self, r1, r2, scene1, scene2):
        loss = torch.nn.BCELoss(reduction='sum')

        l1 = loss(r1, scene1)
        l2 = loss(r2, scene2)

        total_loss = l1 + l2
        return total_loss, l1, l2
