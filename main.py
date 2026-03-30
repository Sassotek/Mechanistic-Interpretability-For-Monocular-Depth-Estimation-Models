from networks.wordepth import WorDepth
import torch
from collections import OrderedDict

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    worDepth_model = WorDepth(pretrained=None).to(device)
    checkpoint_path = "models/nyu_latest"
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

if __name__ == "__main__": main()