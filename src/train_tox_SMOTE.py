'''
Date: 2023-10-03 21:09:14
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:16:17
'''
import os
os.environ["SCIPY_ARRAY_API"] = "1"
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

from molnetpack import MolnetTox_bin
from molnetpack import MolTox_Dataset


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_step(model, device, loader, optimizer, batch_size, num_points) -> tuple[float, float]:
    loss = 0
    accuracy = 0
    criterion = nn.BCEWithLogitsLoss()

    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            print(f"Batch structure: {len(batch)}")
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            print(x.size())
            x = x.permute(0, 2, 1)

            y = y.to(device, dtype=torch.float).view(-1, 1)  # Ensure shape matches (batch_size, 1)

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
            pred_class = (pred >= 0.5).float()
            batch_accuracy = (pred_class == y).float().mean().item()
            accuracy += batch_accuracy

            bar.set_description('Train')
            bar.set_postfix(lr=get_lr(optimizer), loss=batch_loss.item(), acc=batch_accuracy)
            bar.update(1)

    return accuracy / (step + 1), loss / (step + 1)


def eval_step(model: nn.Module, device, loader: DataLoader, batch_size, num_points):
    model.eval()
    accuracy = 0
    total = 0
    val_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            _, x, features, y = batch
            x = x.to(device=device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            y = y.to(device=device, dtype=torch.float).view(batch_size, 1)

            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

            with torch.no_grad():
                logits = model(x, None, idx_base)
                batch_loss = criterion(logits, y)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct = (preds == y).float().sum().item()
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
    parser.add_argument('--train_data', type=str, default='./data/increase_mitochondrial_dysfunction_etkdgv3_train.pkl',
                        help='path to training data (pkl)')
    parser.add_argument('--test_data', type=str, default='./data/increase_mitochondrial_dysfunction_etkdgv3_test.pkl',
                        help='path to test data (pkl)')
    parser.add_argument('--model_config_path', type=str, default='./src/molnetpack/config/molnet_rt.yml',
                        help='path to model and training configuration')
    parser.add_argument('--data_config_path', type=str, default='./src/molnetpack/config/preprocess_etkdgv3.yml',
                        help='path to configuration')
    parser.add_argument('--checkpoint_path', type=str, default='./check_point/molnet_mito_dys_etkdgv3.pt',
                        help='Path to save checkpoint')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--transfer', action='store_true',
                        help='Whether to load the pretrained encoder')
    parser.add_argument('--ex_model_path', type=str, default='',
                        help='Path to export the whole model (structure & weights)')
    parser.add_argument('--validation_only', action='store_true',
                        help='Run validation only without training')
    parser.add_argument('--plot', type=str, default='./plots',
                        help='Directory to save the plot')
    parser.add_argument('--eval_only', action='store_true', help="Only evaluate the model without training")

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random functions')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Enables CUDA training')

    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
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



    X = []
    y = []

    for i in range(len(train_set)):
        _, x_i, _, y_i = train_set[i]

            #print("X shape before:", x_i.shape)
           # print("y shape before:", y_i.shape)

        X.append(x_i.flatten())
        y.append(y_i)

            #print("X shape after flat:", x_i.flatten().shape)
           # print("y shape flat:", y_i.shape)
    X = np.stack(X)
    y = np.stack(y)

    smote = SMOTE(random_state = args.seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)



    # print("X shape:", X_resampled.shape)
    # print("y shape:", y_resampled.shape)
    # sys.exit(0)

    X_tensor = torch.tensor(X_resampled, dtype=torch.float32).view(-1, 300, 21)
    y_tensor = torch.tensor(y_resampled, dtype=torch.float32)

    data_set_smote = TensorDataset(X_tensor, y_tensor)

    train_loader = DataLoader(
        data_set_smote,
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

    model = MolnetTox_bin(config['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{str(model)} #Params: {num_params}')

    if args.eval_only:
        print("Loading trained model for evaluation...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Evaluate on test data
        test_accuracy, test_loss = eval_step(
            model=model,
            device=device,
            loader=valid_loader,
            batch_size=config['train']['batch_size'],
            num_points=config['model']['max_atom_num']
        )

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        sys.exit()

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=10)

    # 4. Train
    best_valid_accuracy = 0
    if args.resume_path != '':
        if args.transfer:
            print("Load the pretrained encoder (freeze the encoder)...")
            checkpoint = torch.load(args.resume_path, map_location=device)['model_state_dict']
            encoder_dict = {}
            for name, param in checkpoint.items():
                if not name.startswith("decoder") and not name.startswith("classifier"):
                    param.requires_grad = False
                    encoder_dict[name] = param
            model.load_state_dict(encoder_dict, strict=False)

        else:
            print("Load the checkpoints...")
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_valid_accuracy = checkpoint['best_val_accuracy']

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
                                               batch_size=config['train']['batch_size'],
                                               num_points=config['model']['max_atom_num'])

        print(
            f"Train: Accuracy: {train_accuracy}, Loss: {train_loss} \nValidation: Accuracy: {valid_accuracy}, Loss: {valid_loss}")

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
            print("Accuracy saved")

        scheduler.step(valid_accuracy)

        print(f'Best accuracy so far: {best_valid_accuracy}')

    x = list(range(1, len(train_losses) + 1))
    print(f"Epochs: {len(x)}, Train Losses: {len(train_losses)}, Valid Losses: {len(valid_losses)}")
    # Training loss
    fig, ax = plt.subplots()
    ax.plot(x, train_losses, linewidth=2.0, label="Training Loss", color="blue")
    ax.plot(x, valid_losses, linewidth=2.0, label="Validation Loss", color="red")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")

    ax.legend(loc="upper left")
    plot_dir = args.plot
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, 'training_vs_validation_loss.png')
    fig.savefig(plot_filename)

    print(f"Plot saved at {plot_filename}")

    if args.ex_model_path != '':
        print('Export the model...')
        model_scripted = torch.jit.script(model)
        model_scripted.save(args.ex_model_path)
        print(f"Model exported to {args.ex_model_path}")
