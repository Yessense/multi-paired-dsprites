import itertools
import operator
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import IterableDataset, DataLoader

modes = ['inference', 'two objects']


class MultiDisDsprites(IterableDataset):
    """IterableDataset with dSprites scenes

    dSprites features:
        Color: white
        Shape: square, ellipse, heart
        Scale: 6 values linearly spaced in [0.5, 1]
        Orientation: 40 values in [0, 2 pi]
        Position X: 32 values in [0, 1]
        Position Y: 32 values in [0, 1]
    """

    def __init__(self,
                 path='./data/dSprites.npz',
                 size: int = 10 ** 6,
                 mode='two objects'):
        self.mode = mode

        # Load zip file
        dataset_zip = np.load(path)

        # Get images and labels
        self.imgs = dataset_zip['imgs']
        self.labels = dataset_zip['latents_classes'][:, 1:]

        # feature info
        self.dsprites_size = len(self.imgs)
        self.size = size
        self.lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
        self.features_count = [3, 6, 40, 32, 32]
        self.features_range = [list(range(i)) for i in self.features_count]
        self.multiplier = list(itertools.accumulate(self.features_count[-1:0:-1], operator.mul))[::-1] + [1]

    def _get_element_pos(self, labels: List[int]) -> int:
        """ Get position of image with `labels` in dataset """
        pos = 0
        for mult, label in zip(self.multiplier, labels):
            pos += mult * label
        return pos

    def __iter__(self):
        return self._sample_generator()

    def _sample_generator(self):
        """Dataset objects generator"""

        if self.mode == 'two objects':
            for i in range(self.size):
                yield self.two_objects_and_scene()
        if self.mode == 'inference':
            for i in range(self.size):
                yield self.inference_sample()
        else:
            raise NameError(f'{self.mode!r} is a wrong mode')

    def _get_pair(self):
        """get random pair of objects that differ only in one feature """
        idx = random.randint(0, self.dsprites_size - 1)

        img = self.imgs[idx]

        pair_img = self._generate_object(img)

        # img -> (1, 64, 64)
        img = np.expand_dims(img, 0)
        # pair_img ->(1, 64, 64)
        pair_img = np.expand_dims(pair_img, 0)
        return img, pair_img

    def _generate_object(self, *scenes) -> ndarray:
        """Find object that will not intersect any scene in `scenes`"""

        while True:
            # select random image
            n = random.randrange(0, self.dsprites_size)
            obj = self.imgs[n]

            # if image intersect scene, try find next
            select_new = False
            for scene in scenes:
                if np.any(scene & obj):
                    select_new = True
                    break
            if not select_new:
                return obj

    def two_objects_and_scene(self):
        """Returns two objects on scene that won't intersect"""
        # emptyscene
        scene = np.zeros((1, 64, 64), dtype=int)

        # Get pair of images that not intersect
        img, pair_img = self._get_pair()

        scene += img + pair_img

        scene = torch.from_numpy(scene).float()
        img = torch.from_numpy(img).float()
        pair_img = torch.from_numpy(pair_img).float()

        return scene, img, pair_img

    def inference_sample(self):
        """Returns two images and a donor"""
        # empty scene where to add objects
        scene = np.zeros((1, 64, 64), dtype=int)

        # store separate objects (1, 64, 64)
        objs = []

        # contains info if it empty image for consistence
        masks = []

        # number of objects on scene
        n_objs = 3
        for i in range(n_objs):
            obj = self._generate_object(scene, scene)
            scene += obj.astype(int)
            objs.append(obj)

        # stack elements into torch tensors
        scene = torch.from_numpy(scene).float()
        image1, donor, image2 = objs
        image1 = torch.from_numpy(image1).float().unsqueeze(0)
        donor = torch.from_numpy(donor).float().unsqueeze(0)
        image2 = torch.from_numpy(image2).float().unsqueeze(0)

        return scene, image1, donor, image2


if __name__ == '__main__':
    # dataset
    mdd = MultiDisDsprites(
        path='/home/yessense/PycharmProjects/multi-dis-dsprites/src/dataset/data/dsprite_train.npz')


    def show_pairs(mdd: MultiDisDsprites, sample_size: int = 5):
        fig, ax = plt.subplots(sample_size, 2)
        for i in range(sample_size):
            img, pair, exchange_labels = mdd._get_pair()
            ax[i, 0].imshow(img.squeeze(0), cmap='gray')
            ax[i, 1].imshow(pair.squeeze(0), cmap='gray')

        plt.show()


    def show_inference_dataset(mdd: MultiDisDsprites, sample_size: int = 5):
        fig, ax = plt.subplots(sample_size, 4)
        for i in range(sample_size):
            scene, image1, donor, image2 = mdd.inference_sample()
            ax[i, 0].imshow(scene.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 1].imshow(image1.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 2].imshow(donor.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 3].imshow(image2.detach().cpu().numpy().squeeze(0), cmap='gray')

        for i in range(sample_size):
            for j in range(4):
                ax[i, j].set_axis_off()

        plt.show()


    def show_training_dataset(mdd: MultiDisDsprites, batch_size: int = 5):
        batch_size = 5
        loader = DataLoader(mdd, batch_size=batch_size)

        for i, batch in enumerate(loader):
            scene, img, pair_img = batch
            if i % 4000 == 0:

                fig, ax = plt.subplots(batch_size, 3, figsize=(5, 5))
                for i in range(batch_size):
                    for j, column in enumerate(batch):
                        ax[i, j].imshow(column[i].detach().cpu().numpy().squeeze(0), cmap='gray')
                        ax[i, j].set_axis_off()

                plt.show()


    # show_inference_dataset(mdd, 5)
    show_training_dataset(mdd, 5)
