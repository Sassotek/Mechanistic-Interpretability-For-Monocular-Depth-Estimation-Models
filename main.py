import argparse
import os
from networks.wordepth import WorDepth
import torch
from collections import OrderedDict
from utils import load_dataset, print_model_sample, NYU_datastet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-eval', '--eval', action='store_true', default=False, help= "Choose to evaluate the model")
    args = parser.parse_args()

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

    dataset = NYU_datastet("nyu_depth_v2_labeled.mat")

    if args.eval:
        #Print a sample 
        print_model_sample(worDepth_model, dataset, device)

        #Evaluation metrics [MSE, absrel, delta]
        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        worDepth_model.eval()
        total_mse = 0.0
        total_absrel = 0.0
        total_delta1 = 0.0
        total_delta2 = 0.0
        total_delta3 = 0.0
        total_valid_pixels = 0

        with torch.no_grad():
            for batch_images, batch_depths in eval_loader:
                inputs = batch_images.to(device)
                gt = batch_depths.to(device)
                
                preds = worDepth_model(inputs, text_feature_list= torch.zeros((inputs.size(0), 1024), dtype=torch.float32).to(device), sample_from_gaussian= False)
                preds = preds.squeeze(1)  
                
                valid_mask = gt > 0
                pred_valid = preds[valid_mask]
                gt_valid = gt[valid_mask]

                n_valid = gt_valid.numel()
                if n_valid == 0:
                    continue
                
                total_valid_pixels += n_valid

                # Accumulate sums (NOT means) for the batch
                total_mse += torch.sum((gt_valid - pred_valid) ** 2).item()
                total_absrel += torch.sum(torch.abs(gt_valid - pred_valid) / gt_valid).item()

                max_ratio = torch.max(gt_valid / pred_valid, pred_valid / gt_valid)
                total_delta1 += torch.sum((max_ratio < 1.25).float()).item()
                total_delta2 += torch.sum((max_ratio < 1.25**2).float()).item()
                total_delta3 += torch.sum((max_ratio < 1.25**3).float()).item()

        # 3. Calculate final metrics across the entire dataset
        if total_valid_pixels > 0:
            final_mse = total_mse / total_valid_pixels
            final_rmse = final_mse ** 0.5
            final_absrel = total_absrel / total_valid_pixels
            final_delta1 = total_delta1 / total_valid_pixels
            final_delta2 = total_delta2 / total_valid_pixels
            final_delta3 = total_delta3 / total_valid_pixels

            print(f"\n--- Final Results ({len(dataset)} images) ---")
            print(f"MSE: {final_mse:.4f}")
            print(f"RMSE: {final_rmse:.4f}")
            print(f"AbsRel: {final_absrel:.4f}")
            print(f"Delta < 1.25: {final_delta1:.4f}")
            print(f"Delta < 1.25^2: {final_delta2:.4f}")
            print(f"Delta < 1.25^3: {final_delta3:.4f}")
            print(f"Total valid pixels evaluated: {total_valid_pixels}")
            
        else:
            print("No valid depth pixels found during evaluation.")

if __name__ == "__main__": main()