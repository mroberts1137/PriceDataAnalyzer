import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from . import utils

class PriceData(Dataset):
    def __init__(self, setname):
        """
        <setname> can be any of: 'train' to specify the training set
                                'val' to specify the validation set
                                'test' to specify the test set"""
        self.setname = setname
        assert setname in ['train', 'test']

        # Define dataset
        overall_dataset_dir = os.path.join(os.path.join(os.getcwd(), 'load_dataset'), 'price_data')
        self.selected_dataset_dir = os.path.join(overall_dataset_dir, setname)

        # E.g. self.all_filenames = ['006.png','007.png','008.png'] when setname=='val'
        self.all_filenames = os.listdir(self.selected_dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(overall_dataset_dir, 'price_targets.csv'), header=0, index_col=0)
        self.label_meanings = self.all_labels.columns.values.tolist()

    def __len__(self):
        """Return the total number of examples in this split, e.g. if
        self.setname=='train' then return the total number of examples
        in the training set"""
        return len(self.all_filenames)

    def __getitem__(self, idx):
        """Return the example at index [idx]. The example is a dict with keys
        'data' (value: Tensor for an RGB image) and 'label' (value: multi-hot
        vector as Torch tensor of gr truth class labels)."""
        selected_filename = self.all_filenames[idx]
        price_data = np.read_csv(os.path.join(self.selected_dataset_dir, selected_filename))

        # convert image to Tensor and normalize
        #image = utils.to_tensor_and_normalize(imagepil)

        # load label
        label = torch.Tensor(self.all_labels.loc[selected_filename, :].values)

        sample = {'data': price_data,  # preprocessed image, for input into NN
                  'label': label,
                  'img_idx': idx}
        return sample
