__author__ = 'marvinler'

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, slides_folders, model_input_size, is_training, max_bag_size, logger, max_dataset_size=None,
                 with_data_augmentation=True, seed=123, normalization_mean=None, normalization_std=None):
        """
        :param slides_folders: list of abs paths of slide folder (which should contains images, summary/label/percent
            files
        :param model_input_size: expected model input size (for cropping)
        :param is_training: True if is training, else False (for data augmentation)
        :param max_bag_size: maximum number of instances to be returned per bag
        """

        def verify_slide_folder_exists(slide_folder):
            if not os.path.exists(slide_folder):
                raise FileExistsError('parent dataset folder %s does not exist' % slide_folder)

        list(map(verify_slide_folder_exists, slides_folders))

        self.slides_folders = np.asarray(slides_folders)
        self.model_input_size = model_input_size
        self.max_bag_size = max_bag_size
        self.max_dataset_size = max_dataset_size

        self.is_training = is_training

        self.logger = logger

        self.slides_ids = []  # ids slides
        self.slides_labels = []  # raw str labels
        self.slides_summaries = []  # list of all initial tiles of slides
        self.slides_cases = []  # list of all cases IDs
        self.slides_images_filepaths = []  # list of all in-dataset tilespaths of slides

        self.with_data_augmentation = with_data_augmentation
        normalization_mean = (0, 0, 0) if normalization_mean is None else normalization_mean
        normalization_std = (1, 1, 1) if normalization_std is None else normalization_std
        self.transform = self._define_data_transforms(normalization_mean, normalization_std)

        self.seed = seed

        slides_ids, slides_labels, slides_summaries, slides_cases, slides_images_filepaths = self.load_data()
        self.slides_ids = slides_ids
        self.slides_labels = slides_labels
        self.slides_summaries = slides_summaries
        self.slides_cases = slides_cases
        self.slides_images_filepaths = slides_images_filepaths

        assert len(self.slides_ids) == len(self.slides_labels) == len(self.slides_summaries) == \
               len(self.slides_images_filepaths), 'mismatch in slides containers lengths %s' % (
            ' '.join(str(len(l)) for l in [self.slides_ids, self.slides_labels, self.slides_summaries,
                                           self.slides_images_filepaths]))

        self.retrieve_tiles_ids_with_images = False  # True will return bag of images and associated tiles ids

    def _define_data_transforms(self, mean, std):
        if self.with_data_augmentation:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def load_data(self):
        slides_ids, slides_labels, slides_summaries, slides_cases, slides_images_filepaths = [], [], [], [], []

        # Name of expected non-image files for all slides folders
        label_filename = 'label.txt'
        case_id_filename = 'case_id.txt'
        summary_filename = 'summary.txt'

        # Seek all slides folders, and load static data including list of tiles filepaths and bag label
        for i, slide_folder in enumerate(tqdm(self.slides_folders)):
            if self.max_dataset_size is not None and i + 1 > self.max_dataset_size:
                break

            all_slide_files = list(filter(lambda f: os.path.isfile(os.path.join(slide_folder, f)),
                                          os.listdir(slide_folder)))

            # Seek and save label, case_id and summary files: expects 1 and only 1 for each
            for data_filename in [label_filename, case_id_filename, summary_filename]:
                assert sum([f == data_filename for f in all_slide_files]) == 1, \
                    'slide %s: found %d files for %s, expected 1' % (slide_folder,
                                                                     sum([f == data_filename for f in
                                                                          all_slide_files], ),
                                                                     data_filename)

            label_file = os.path.join(slide_folder, [f for f in all_slide_files if f == label_filename][0])
            case_id_file = os.path.join(slide_folder, [f for f in all_slide_files if f == case_id_filename][0])
            summary_file = os.path.join(slide_folder, [f for f in all_slide_files if f == summary_filename][0])
            with open(label_file, 'r') as f:
                slide_label = int(f.read())
            with open(case_id_file, 'r') as f:
                slide_case_id = f.read()
            with open(summary_file, 'r') as f:
                slide_original_tiles = f.read().splitlines()

            # Seek all filtered images of slide (not-background images)
            slide_images_filenames = list(filter(lambda f: f.endswith(('.jpeg', '.jpg', '.png')), all_slide_files))

            if len(slide_images_filenames) == 0:
                self.logger.warning('Discarding slide %s of class %d because there are no images' %
                                    (slide_folder, slide_label))
                continue

            # Save data
            slides_ids.append(os.path.basename(slide_folder))
            slides_labels.append(slide_label)
            slides_summaries.append(slide_original_tiles)
            slides_cases.append(slide_case_id)
            slides_images_filepaths.append(
                list(map(lambda f: os.path.abspath(os.path.join(slide_folder, f)), slide_images_filenames)))

        slides_ids = np.asarray(slides_ids)
        slides_labels = np.asarray(slides_labels)

        return slides_ids, slides_labels, slides_summaries, slides_cases, slides_images_filepaths

    def show_bag(self, bag_idx, savefolder=None):
        """ Plot/save tiles sampled from the slide of provided index """
        bag = self._get_slide_instances(bag_idx)
        bag_label = self.slides_labels[bag_idx]
        tr = transforms.ToTensor()
        bag = [tr(b) for b in bag]
        imgs = make_grid(bag)

        npimgs = imgs.numpy()
        plt.imshow(np.transpose(npimgs, (1, 2, 0)), interpolation='nearest')
        plt.title('Bag label: %s | %d instances' % (bag_label, len(bag)))
        if savefolder is not None:
            plt.savefig(os.path.join(savefolder, 'show_' + str(bag_idx) + '.png'), dpi=1000)
        else:
            plt.show()

    def _get_slide_instances(self, item):
        """ Memory load all tiles or randomly sampled tiles from slide of specified index """
        slide_images_filepaths = self.slides_images_filepaths[item]

        # Randomly sample the specified max number of tiles from the slide with replacement
        if self.max_bag_size is not None:
            slide_images_filepaths = random.choices(slide_images_filepaths, k=self.max_bag_size)

        # Load images
        bag_images = [pil_loader(slide_image_filepath) for slide_image_filepath in slide_images_filepaths]

        if self.retrieve_tiles_ids_with_images:
            # return bag of images as well as the associated ids of the tiles
            return bag_images, list(map(os.path.basename, slide_images_filepaths)), self.slides_summaries[item]
        return bag_images

    def __getitem__(self, item):
        if not self.retrieve_tiles_ids_with_images:
            slide_instances = self._get_slide_instances(item)
            slide_instances = torch.stack([self.transform(instance) for instance in slide_instances])
            slide_label = self.slides_labels[item]
            return slide_instances, slide_label

        slide_instances, tiles_ids, slide_summary = self._get_slide_instances(item)
        slide_instances = torch.stack([self.transform(instance) for instance in slide_instances])
        slide_label = self.slides_labels[item]
        return slide_instances, slide_label, tiles_ids, slide_summary

    def __len__(self):
        return len(self.slides_labels)
