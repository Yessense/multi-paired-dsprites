from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import sys

sys.path.append("..")

from pytorch_lightning.loggers import WandbLogger

from src.dataset import dataset
import scene_vae
import pytorch_lightning as pl
from argparse import ArgumentParser

wandb_logger = WandbLogger(project='multi-paired-dsprites-vae', log_model=True)

# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--dataset_size", type=int, default=10 ** 6)
program_parser.add_argument("--batch_size", type=int, default=128)
program_parser.add_argument("--dataset_mode", type=str, default='two objects')
program_parser.add_argument("--dataset_path", type=str,
                            default='/home/yessense/PycharmProjects/Multi-paired-dsprites/src/dataset/data/dsprites.npz')

# add model specific args
parser = scene_vae.MultiPairedDspritesVAE.add_model_specific_args(parent_parser=parser)

# add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

iterable_dataset = dataset.MultiDisDsprites(size=args.dataset_size, path=args.dataset_path, mode=)
loader = DataLoader(iterable_dataset, batch_size=args.batch_size, num_workers=1)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)
autoencoder = scene_vae.MultiPairedDspritesVAE(**dict_args)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'BCE reconstruct'

# early stop
patience = 5
early_stop_callback = EarlyStopping(monitor=monitor, patience=patience)

# checkpoint
save_top_k = 2
save_weights_only = True

checkpoint_callback = ModelCheckpoint(monitor=monitor,
                                      save_weights_only=save_weights_only,
                                      save_top_k=save_top_k)

callbacks = [
    checkpoint_callback,
    # early_stop_callback,
]

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

# trainer parameters
profiler = 'simple'  # 'simple'/'advanced'/None
gpus = [0]

# trainer
trainer = pl.Trainer(gpus=gpus,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     limit_val_batches=0.0,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     # checkpoint_callback=checkpoint_callback)
                     )
trainer.fit(autoencoder, loader)
