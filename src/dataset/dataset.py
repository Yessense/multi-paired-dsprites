import itertools
import operator
import random
from typing import List, Tuple, Sequence

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
                 path: str = './data/dSprites.npz',
                 size: int = 10 ** 6,
                 mode: str = 'two objects',
                 max_exchanges: int = 1):
        self.max_exchanges = max_exchanges
        self.mode = mode

        # Load npz numpy archive
        dataset_zip = np.load(path)

        # Images: numpy array -> (737280, 64, 64)
        self.imgs = dataset_zip['imgs']

        # Labels: numpy array -> (737280, 5)
        # Each column contains int value in range of `features_count`
        self.labels = dataset_zip['latents_classes'][:, 1:]

        self.size = size

        # ----------------------------------------
        # features info
        # ----------------------------------------

        # Size of dataset (737280)
        self.dsprites_size = len(self.imgs)

        # List of feature names
        self.feature_names: Tuple[str, ...] = ('shape', 'scale', 'orientation', 'posX', 'posY')

        # Feature numbers
        self.features_list: List[int] = list(range(len(self.feature_names)))

        # Count each feature counts
        self.features_count = [3, 6, 40, 32, 32]

        # Getting multipler for each feature position
        self.features_range = [list(range(i)) for i in self.features_count]
        self.multiplier = list(itertools.accumulate(self.features_count[-1:0:-1], operator.mul))[::-1] + [1]

    def _get_element_pos(self, labels: List[int]) -> int:
        """ Get position of image with `labels` in dataset """
        pos = 0
        for mult, label in zip(self.multiplier, labels):
            pos += mult * label
        return pos

    def __iter__(self):
        if self.mode == 'two objects':
            return self._sample_generator(gen_object=self.two_objects_and_scene)
        elif self.mode == 'inference':
            return self._sample_generator(gen_object=self.inference_sample)
        elif self.mode == 'exchange':
            return self._sample_generator(gen_object=self.exchange_sample)
        else:
            raise NameError("Wrong mode")

    def _sample_generator(self, gen_object):
        """Dataset objects generator"""
        for i in range(self.size):
            yield gen_object()

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
            obj = self._generate_object(scene)
            scene += obj.astype(int)
            objs.append(obj)

        # stack elements into torch tensors
        scene = torch.from_numpy(scene).float()
        image1, donor, image2 = objs
        image1 = torch.from_numpy(image1).float().unsqueeze(0)
        donor = torch.from_numpy(donor).float().unsqueeze(0)
        image2 = torch.from_numpy(image2).float().unsqueeze(0)

        return scene, image1, donor, image2

    def exchange_sample(self):
        """Generate 4 objects on scene: img1, img2, donor, pair img to img1"""

        # Empty scene
        scene = np.zeros((1, 64, 64), dtype=int)

        # Generate img1
        n = random.randrange(0, self.dsprites_size)

        img = self.imgs[n]
        labels = self.labels[n]
        scene += img

        # Generate img2 and donor
        objs = []
        n_objs = 2

        for i in range(n_objs):
            obj = self._generate_object(scene)
            objs.append(obj)

        donor, img2 = objs

        # Generate pair img
        # Choose number of exchanges
        n_exchanges = random.randrange(1, self.max_exchanges + 1)

        # select features that will be exchanged
        exchanges = random.sample(population=self.features_list, k=n_exchanges)

        exchange_labels = np.zeros_like(labels, dtype=bool)
        pair_img_labels = labels[:]

        for feature_type in exchanges:
            # Find other feature and add his number to pair_img_labels
            exchange_labels[feature_type] = True

            other_feature = random.choice(self.features_range[feature_type])

            while other_feature == labels[feature_type]:
                other_feature = random.choice(self.features_range[feature_type])

            pair_img_labels[feature_type] = other_feature

        pair_idx: int = self._get_element_pos(pair_img_labels)
        pair_img = self.imgs[pair_idx]

        scene = img2.astype(bool) | pair_img.astype(bool)

        img = torch.from_numpy(img).float().unsqueeze(0)
        img2 = torch.from_numpy(img2).float().unsqueeze(0)
        donor = torch.from_numpy(donor).float().unsqueeze(0)
        pair_img = torch.from_numpy(pair_img).float().unsqueeze(0)
        scene = torch.from_numpy(scene).float().unsqueeze(0)
        exchange_labels = torch.from_numpy(exchange_labels).unsqueeze(-1)

        return img, img2, donor, pair_img, scene, exchange_labels


if __name__ == '__main__':
    # dataset
    mdd = MultiDisDsprites(
        path='/home/yessense/PycharmProjects/multi-dis-dsprites/src/dataset/data/dsprite_train.npz', mode='exchange',
        max_exchanges=5)


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


    def plot_list(suptitle: str, titles: Sequence[str], images: List):
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


    def show_exchange_dataset(mdd: MultiDisDsprites):
        short_y_labels = {0: 'Shape', 1: 'Scale', 2: 'Orientation', 3: 'X', 4: 'Y'}

        def make_y_label_name(idx):
            names = [short_y_labels[i] for i, value in enumerate(idx) if value]
            name = ", ".join(names)
            return name

        loader = DataLoader(mdd, batch_size=1)

        batch = next(iter(loader))

        img, img2, donor, pair_img, scene, exchange_labels = batch
        indices = exchange_labels.squeeze().numpy()

        titles = ['Img', 'Img2', 'Donor', 'Pair img', 'Scene']

        plot_list(f'Exchanges dataset (exchanges = {make_y_label_name(indices)})', titles=titles,
                  images=[img, pair_img, img2, donor, scene])


    # show_inference_dataset(mdd, 5)
    # show_training_dataset(mdd, 5)

    show_exchange_dataset(mdd)
    show_exchange_dataset(mdd)
    show_exchange_dataset(mdd)
