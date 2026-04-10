import h5py
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt 

def load_dataset(dataset_path):
    print(f"Loading dataset from {dataset_path}...")
    dataset_file = h5py.File(dataset_path, 'r')
    images = np.array(dataset_file['images'])
    depths = np.array(dataset_file['depths'])

    images = images.transpose((0, 1, 3, 2))
    depths = depths.transpose((0, 2, 1))

    print(f"Dataset loaded: {images.shape[0]} samples, image shape: {images.shape[1:]}, depth shape: {depths.shape[1:]}")
    return torch.from_numpy(images).float(), torch.from_numpy(depths).float()

def print_model_sample(model, dataset, device):
    model.eval()
    text_features = torch.zeros((1, 1024), dtype=torch.float32).to(device)
    with torch.no_grad():
        output_dir = "./outputs"
        i = random.randint(0, len(dataset) - 1)
        image, depth = dataset[i]
        image = image.unsqueeze(0).to(device)
        image = image / 255.0
        output = model(image, text_feature_list= text_features, sample_from_gaussian= False)
        predicted_depth = output 

        depth_map = predicted_depth.squeeze().cpu().numpy()
        rgb_image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        rgb_image = np.clip(rgb_image, 0, 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original RGB Image")
        axes[0].axis('off') 
        im = axes[1].imshow(depth_map, cmap='inferno')
        axes[1].set_title("Predicted Depth")
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], label='Depth', fraction=0.046, pad=0.04)
        plt.tight_layout()
        save_path = os.path.join(output_dir, "side_by_side_prediction.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Saved depth map to {save_path}")

class NYU_datastet(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.images, self.depths = load_dataset(dataset_path)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx] / 255.0
        depth = self.depths[idx]
        return image, depth