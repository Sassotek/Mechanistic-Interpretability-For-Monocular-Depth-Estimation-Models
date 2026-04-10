import h5py
import numpy as np
import torch

def load_dataset(dataset_path):
    print(f"Loading dataset from {dataset_path}...")
    dataset_file = h5py.File(dataset_path, 'r')
    images = np.array(dataset_file['images'])
    depths = np.array(dataset_file['depths'])

    images = images.transpose((0, 1, 3, 2))
    depths = depths.transpose((0, 2, 1))

    print(f"Dataset loaded: {images.shape[0]} samples, image shape: {images.shape[1:]}, depth shape: {depths.shape[1:]}")
    return torch.from_numpy(images).float(), torch.from_numpy(depths).float()