'''
Date: 2023-10-03 21:09:14
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:16:17
'''
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
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


def train_step(model, device, loader, optimizer, batch_size, num_points) -> tuple[int, int]:
    loss = 0
    accuracy = 0
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification

    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            print(f"Batch structure: {len(batch)}")
            _, x, features, y = batch
            x = x.to(device=device, dtype=torch.float)
            print(x.size())
            x = x.permute(0, 2, 1)

            y = y.to(device=device, dtype=torch.long).view(batch_size)

            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

            optimizer.zero_grad()
            model.train()
            pred = model(x, None, idx_base)

            # Compute loss
            batch_loss = criterion(pred, y)
            loss += int(batch_loss)

            batch_loss.backward()
            optimizer.step()

            # Compute accuracy
            pred_class = pred.argmax(dim=1)
            batch_accuracy = (pred_class == y).float().mean().item()
            accuracy += batch_accuracy

            bar.set_description('Train')
            bar.set_postfix(lr=get_lr(optimizer), loss=batch_loss.item(), acc=batch_accuracy)
            bar.update(1)

    return accuracy / (step + 1), loss / (step + 1)


# def train_step(model, device, loader, optimizer, batch_size, num_points):
#     accuracy = 0
#     with tqdm(total=len(loader)) as bar:
#         for step, batch in enumerate(loader):
#             print(f"Batch structure: {len(batch)}")
#             _, x, features, y = batch
#             x = x.to(device=device, dtype=torch.float)
#             print(x.size())
#             x = x.permute(0, 2, 1)
#             y = y.to(device=device, dtype=torch.float).view(-1, 1)  # Ensure shape is [batch_size, 1]
#
#             idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
#
#             optimizer.zero_grad()
#             model.train()
#             pred = model(x, None, idx_base)  # `pred` should be logits, not probabilities
#
#             # Use BCEWithLogitsLoss for numerical stability
#             loss_fn = nn.BCEWithLogitsLoss()
#             loss = loss_fn(pred, y)
#
#             loss.backward()
#             optimizer.step()
#
#             # Convert logits to probabilities for accuracy calculation
#             pred_binary = torch.sigmoid(pred).round()  # Converts logits to 0 or 1
#             accuracy += (pred_binary == y).float().mean().item()  # Compute accuracy
#
#             bar.set_description('Train')
#             bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
#             bar.update(1)
#
#     return accuracy / (step + 1)  # Return average accuracy over all batches

def eval_step(model: nn.Module, device, loader: DataLoader, batch_size, num_points):
    model.eval()
    accuracy = 0
    total = 0
    val_loss = 0
    criterion = nn.CrossEntropyLoss()

    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, features, y = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            y = y.to(device=device, dtype=torch.long)
            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

            with torch.no_grad():
                pred = model(x, None, idx_base)
                batch_loss = criterion(pred, y)

            _, predicted = torch.max(pred, 1)

            correct = (predicted == y).sum().item()
            accuracy += correct
            total += y.size(0)
            val_loss += batch_loss.item()

            bar.set_description('Eval')
            bar.update(1)

    return accuracy / total, val_loss / (step + 1)


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
    parser.add_argument('--checkpoint_path', type=str, default='./check_point/molnet_rt_etkdgv3.pt',
                        help='Path to save checkpoint')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--transfer', action='store_true',
                        help='Whether to load the pretrained encoder')
    parser.add_argument('--ex_model_path', type=str, default='',
                        help='Path to export the whole model (structure & weights)')
    parser.add_argument('--validation_only', action='store_true',
                        help = 'Run validation only without training')

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
    assert config['model']['batch_size'] == config['train'][
        'batch_size'], "Batch size should be the same in model and training configuration"

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
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    print(f'Device: {device}')

    model = MolNet_tox(config['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{str(model)} #Params: {num_params}')

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=10)  # Accuracy should improve

    # Load checkpoint if applicable
    if args.resume_path != '':
        print("Loading checkpoint...")
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_valid_accuracy = checkpoint['best_val_mae']
    else:
        best_valid_accuracy = 0

    # Handle transfer learning
    if args.transfer and args.resume_path != '':
        print("Freezing encoder layers...")
        for name, param in model.named_parameters():
            if not name.startswith("decoder"):
                param.requires_grad = False  # Freeze encoder

    # Ensure checkpoint path exists
    if args.checkpoint_path != '':
        checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    early_stop_step = 30
    early_stop_patience = 0

    train_losses = []
    valid_losses = []


    for epoch in range(1, config['train']['epochs'] + 1):
        print("\n=====Epoch {}".format(epoch))
        train_accuracy, train_loss = train_step(model, device, train_loader, optimizer,
                                                batch_size=config['train']['batch_size'],
                                                num_points=config['model']['max_atom_num'])
        valid_accuracy, valid_loss = eval_step(model, device, valid_loader,
                                   batch_size=config['train']['batch_size'], num_points=config['model']['max_atom_num'])

        print(f"Train: Accuracy: {train_accuracy}, Loss: {train_loss} \nValidation: Accuracy: {valid_accuracy}, Loss: {valid_loss}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Update best accuracy
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            if args.checkpoint_path != '':
                print('Saving checkpoint...')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_mae': best_valid_accuracy,
                    'num_params': num_params
                }
                torch.save(checkpoint, args.checkpoint_path)
            early_stop_patience = 0
            print('Early stop patience reset')
        else:
            early_stop_patience += 1
            print(f'Early stop count: {early_stop_patience}/{early_stop_step}')

        # Reduce LR if validation accuracy does not improve
        scheduler.step(valid_accuracy)

        print(f'Best accuracy so far: {best_valid_accuracy}')

        # # method 1
        # for epoch in range(1, config['train']['epochs'] + 1):
        #     x.append(epoch)
        # # method 2
        # x.extend(epoch in range(1, config['train']['epochs'] + 1))
        # # method 3
        # x.extend(range(1, config['train']['epochs'] + 1))
        # method 4

        if early_stop_patience >= early_stop_step:
            print('Early stop!')
            break

    # Load Best Saved Model & Validate Again
    print("Loading the bet saved model for final validation....")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Final validation with the loaded model
    final_valid_accuracy = eval_step(model, device, valid_loader,
                                     batch_size = config['train']['batch_size'],
                                     num_points = config['model']['max_atom_num'])

    print(f"Final Validation Accuracy with Best Model: {final_valid_accuracy}")

    x = list(range(1, config['train']['epochs'] + 1))

    # Training loss
    fig, ax = plt.subplots()
    ax.plot(x, train_losses, linewidth = 2.0, label = "Training Loss", color = "blue")
    ax.plot(x, valid_losses, linewidth = 2.0, label = "Validation Loss", color = "red")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))
    plt.show()

    if args.ex_model_path != '':
        print('Export the model...')
        model_scripted = torch.jit.script(model)
        model_scripted.save(args.ex_model_path)
        print(f"Model exported to {args.ex_model_path}")

    if args.validation_only:
        print("Running validation...")
        print(f"Loading model from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        final_validation = eval_step(model, device, valid_loader,
                                     batch_size = config['train']['batch_size'],
                                     num_points = config['model']['max_atom_num'])
        print(f"Final Validation Accuracy: {final_valid_accuracy}")

        exit()
# python ./src/train_tox.py --train_data ./data/cardio_toxicity_etkdgv3_train.pkl \
# --test_data ./data/cardio_toxicity_etkdgv3_test.pkl \
# --model_config_path ./src/molnetpack/config/molnet_rt.yml \
# --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
# --checkpoint_path ./check_point/molnet_rt_etkdgv3.pt \
# --ex_model_path ./check_point/

# For validation only
# python ./src/train_tox.py --validate_only \
# --test_data ./data/cardio_toxicity_etkdgv3_test.pkl \
# --model_config_path ./src/molnetpack/config/molnet_rt.yml \
# --data_config_path ./src/molnetpack/config/preprocess_etkdgv3.yml \
# --checkpoint_path ./check_point/molnet_rt_etkdgv3.pt
