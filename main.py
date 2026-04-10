import os
from networks.wordepth import WorDepth
import torch
from collections import OrderedDict
from utils import load_dataset
import random
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    worDepth_model = WorDepth(pretrained=None).to(device)
    checkpoint_path = "models/nyu_latest"
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint 
    else:
        state_dict = checkpoint

    #since the model was trained with multiple GPUs
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    worDepth_model.load_state_dict(new_state_dict)

    print("Model loaded")

    images , depths = load_dataset("nyu_depth_v2_labeled.mat")
    text_features = torch.zeros((1, 1024), dtype=torch.float32).to(device)

    worDepth_model.eval()
    with torch.no_grad():
        output_dir = "./outputs"
        i = random.randint(0, images.shape[0] - 1)
        image = images[i].unsqueeze(0).to(device)
        image = image / 255.0
        output = worDepth_model(image, text_feature_list= text_features, sample_from_gaussian= False)
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

if __name__ == "__main__": main()