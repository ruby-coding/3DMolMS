'''
Date: 2023-10-03 21:09:14
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:16:17
'''
import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from molnetpack import MolNet_tox
from molnetpack import MolTox_Dataset


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def train_step(model, device, loader, optimizer, batch_size, num_points): 
    accuracy = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            print(f"Batch structure: {len(batch)}")
            _, x, features, y = batch
            x = x.to(device=device, dtype=torch.float)
            print(x.size())
            x = x.permute(0, 2, 1)
            y = y.to(device=device, dtype=torch.float).view(-1, 1)  # Ensure shape is [batch_size, 1]
            
            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

            optimizer.zero_grad()
            model.train()
            pred = model(x, None, idx_base)  # `pred` should be logits, not probabilities
            
            # Use BCEWithLogitsLoss for numerical stability
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(pred, y)  

            loss.backward()
            optimizer.step()

            # Convert logits to probabilities for accuracy calculation
            pred_binary = torch.sigmoid(pred).round()  # Converts logits to 0 or 1
            accuracy += (pred_binary == y).float().mean().item()  # Compute accuracy

            bar.set_description('Train')
            bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
            bar.update(1)

    return accuracy / (step + 1)  # Return average accuracy over all batches

# def train_step(model, device, loader, optimizer, batch_size, num_points): 
# 	accuracy = 0
# 	with tqdm(total=len(loader)) as bar:
# 		for step, batch in enumerate(loader):
# 			print(f"Batch structure: {len(batch)}")
# 			_, x, features, y = batch
# 			x = x.to(device=device, dtype=torch.float)
# 			print(x.size())
# 			x = x.permute(0, 2, 1)
# 			y = y.to(device=device, dtype=torch.float)
# 			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

# 			optimizer.zero_grad()
# 			model.train()
# 			pred = model(x, None, idx_base) 
# 			loss_fn = nn.BCEWithLogitsLoss()
#             loss = loss_fn(pred, y.float())  # Ensure y is float type
# 			loss.backward()

# 			bar.set_description('Train')
# 			bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
# 			bar.update(1)

# 			optimizer.step()

# 			accuracy += torch.abs(pred - y).mean().item()
# 	return accuracy / (step + 1)

def eval_step(model: nn.Module, device, loader, batch_size, num_points):
    model.eval()
    accuracy = 0
    total = 0
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, features, y = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)  # Adjust dimensions for model input
            y = y.to(device=device, dtype=torch.long)  # Ensure labels are integers for classification
            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

            with torch.no_grad():
                pred = model(x, None, idx_base)

            # Convert predictions to class indices (e.g., for multi-class classification)
            _, predicted = torch.max(pred, 1)  # Get the class with the highest score

            # Update accuracy
            correct = (predicted == y).sum().item()
            accuracy += correct
            total += y.size(0)  # Total number of samples

            bar.set_description('Eval')
            bar.update(1)

    # Compute overall accuracy
    return accuracy / total


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# python ./src/train_rt.py --train_data ./data/origin/metlin_etkdgv3_train.pkl\
# --test_data ./data/origin/metlin_etkdgv3_test.pkl \
# --model_config_path ./src/molnetpack/config/molnet_rt.yml \
# --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
# --checkpoint_path ./check_point/molnet_rt_etkdgv3_tl.pt \
# --transfer \
# --resume_path ./check_point/molnet_qtof_etkdgv3.pt


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Molecular Retention Time Prediction (Train)')
    parser.add_argument('--train_data', type=str, default='./data/cardio_toxicity_etkdgv3_train.pkl',
                        help='path to training data (pkl)')
    parser.add_argument('--test_data', type=str, default='./data/cardio_toxicity_etkdgv3_test.pkl',
                        help='path to test data (pkl)')
    parser.add_argument('--model_config_path', type=str, default='./src/molnetpack/config/molnet_rt.yml',
                        help='path to model and training configuration')
    parser.add_argument('--data_config_path', type=str, default='./src/molnetpack/config/preprocess_etkdgv3.yml',
                        help='path to configuration')
    parser.add_argument('--checkpoint_path', type=str, default = './check_point/molnet_rt_etkdgv3.pt',
                        help='Path to save checkpoint')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--transfer', action='store_true',
                        help='Whether to load the pretrained encoder')
    parser.add_argument('--ex_model_path', type=str, default='',
                        help='Path to export the whole model (structure & weights)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random functions')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Enables CUDA training')
    args = parser.parse_args()

    init_random_seed(args.seed)
    with open(args.model_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('Load the model & training configuration from {}'.format(args.model_config_path))
    # configuration check
    assert config['model']['batch_size'] == config['train']['batch_size'], "Batch size should be the same in model and training configuration"

    # 1. Data
    train_set = MolTox_Dataset(args.train_data)
    train_loader = DataLoader(
                    train_set,
                    batch_size=config['train']['batch_size'],
                    shuffle=True,
                    num_workers=config['train']['num_workers'],
                    drop_last=True)
    valid_set = MolTox_Dataset(args.test_data)
    valid_loader = DataLoader(
                    valid_set,
                    batch_size=config['train']['batch_size'],
                    shuffle=True,
                    num_workers=config['train']['num_workers'],
                    drop_last=True)

    # 2. Model
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    print(f'Device: {device}')

    model = MolNet_tox(config['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{str(model)} #Params: {num_params}')

    # 3. Train
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    if args.transfer and args.resume_path != '':
        print("Load the pretrained encoder (freeze the encoder)...")
        state_dict = torch.load(args.resume_path, map_location=device, weights_only=True)['model_state_dict']
        encoder_dict = {}
        for name, param in state_dict.items():
            if not name.startswith("decoder"):
                param.requires_grad = False # freeze the encoder
                encoder_dict[name] = param
        model.load_state_dict(encoder_dict, strict=False)
    elif args.resume_path != '':
        print("Load the checkpoints...")
        model.load_state_dict(torch.load(args.resume_path, map_location=device, weights_only=True)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.resume_path, map_location=device, weights_only=True)['optimizer_state_dict'])
        scheduler.load_state_dict(torch.load(args.resume_path, map_location=device, weights_only=True)['scheduler_state_dict'])
        best_valid_accuracy = torch.load(args.resume_path, weights_only=True)['best_val_accuracy']

    if args.checkpoint_path != '':
        checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
        os.makedirs(checkpoint_dir, exist_ok = True)

    best_valid_accuracy = 0
    early_stop_step = 30
    early_stop_patience = 0
    for epoch in range(1, config['train']['epochs'] + 1):
        print("\n=====Epoch {}".format(epoch))
        train_accuracy = train_step(model, device, train_loader, optimizer,
                                batch_size=config['train']['batch_size'], num_points=config['model']['max_atom_num'])
        valid_accuracy = eval_step(model, device, valid_loader,
                                batch_size=config['train']['batch_size'], num_points=config['model']['max_atom_num'])
        print("Train: Accuracy: {}, \nValidation: Accuracy: {}".format(train_accuracy, valid_accuracy))



        if valid_accuracy < best_valid_accuracy:
            best_valid_accuracy = valid_accuracy

            if args.checkpoint_path != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_accuracy, 'num_params': num_params}
                torch.save(checkpoint, args.checkpoint_path)

            early_stop_patience = 0
            print('Early stop patience reset')
        else:
            early_stop_patience += 1
            print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

        # scheduler.step()
        scheduler.step(valid_accuracy) # ReduceLROnPlateau
        print(f'Best accuracy so far: {best_valid_accuracy}')

        if early_stop_patience == early_stop_step:
            print('Early stop!')
            break

    if args.ex_model_path != '': # export the model
        print('Export the model...')
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(args.ex_model_path) # Save


# python ./src/train_tox.py --train_data ./data/cardio_toxicity_etkdgv3_train.pkl \
# --test_data ./data/cardio_toxicity_etkdgv3_test.pkl \
# --model_config_path ./src/molnetpack/config/molnet_rt.yml \
# --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
# --checkpoint_path ./check_point/molnet_rt_etkdgv3.pt \
# --ex_model_path ./check_point/