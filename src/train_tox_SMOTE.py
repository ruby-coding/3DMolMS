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
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

from molnetpack import MolnetTox_bin
from molnetpack import MolTox_Dataset



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def random_split_data(data_df, test_ratio=0.2, seed=0):
    np.random.seed(seed)
    smiles_list = data_df['smiles'].drop_duplicates().tolist()
    test_smiles = np.random.choice(smiles_list, int(len(smiles_list) * test_ratio), replace=False)

    test_df = data_df[data_df['smiles'].isin(test_smiles)].reset_index(drop=True)
    train_df = data_df[~data_df['smiles'].isin(test_smiles)].reset_index(drop=True)

    return train_df, test_df


def train_step(model, device, loader, optimizer, batch_size, num_points) -> tuple[float, float]:
    total_loss = 0
    total_accuracy = 0
    criterion = nn.BCEWithLogitsLoss()

    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            #print(f"Batch structure: {len(batch)}")
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            #print(x.size())
            x = x.permute(0, 2, 1)

            y = y.to(device, dtype=torch.float).view(-1, 1)  # Ensure shape matches (batch_size, 1)

            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

            optimizer.zero_grad()
            model.train()
            pred = model(x, None, idx_base)

            # Compute loss
            batch_loss = criterion(pred, y)
            total_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            # Compute accuracy
            pred_class = (pred >= 0.5).float()
            batch_accuracy = (pred_class == y).float().mean().item()
            total_accuracy += batch_accuracy

            bar.set_description('Train')
            bar.set_postfix(lr=get_lr(optimizer), loss=batch_loss.item(), acc=batch_accuracy)
            bar.update(1)

    return total_accuracy / (step + 1), total_loss / (step + 1)



def evaluate_model_metrics(model, device, loader: DataLoader, num_points: int, return_preds_targets=False):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            _, x, _, y = batch
            x = x.to(device).float().permute(0, 2, 1)
            y = y.to(device).float().view(-1)

            idx_base = torch.arange(0, x.size(0), device=device).view(-1, 1, 1) * num_points
            outputs = model(x, None, idx_base).squeeze()

            # Compute batch loss on raw logits
            batch_loss = criterion(outputs, y)
            total_loss += batch_loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Loss": total_loss / len(loader)
    }

    if return_preds_targets:
        return metrics, y_true, y_pred
    else:
        return metrics


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
    parser.add_argument('--checkpoint_path', type=str, default='./check_point/(0401)molnet_mito_dys1_etkdgv3.pt',
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
    parser.add_argument('--eval_only_train', action='store_true', help="Only evaluate the model without training")
    parser.add_argument('--eval_only_metrics', action='store_true', help="Only evaluate the model with metrics without training")
    parser.add_argument('--plot_confusion_matrix', action='store_true', help='Flag to plot confusion matrix')


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

    y_labels = y.astype(int) if isinstance(y, np.ndarray) else np.array(y).astype(int)

    # print("🔍 Original class distribution:", Counter(y_labels))


    smote = SMOTE(random_state = args.seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # print("✅ Resampled class distribution:", Counter(y_resampled.astype(int)))
    #Plot PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_resampled)
    # original_class_1_count = sum(y == 1)
    # smote_added_count = sum(y_resampled == 1) - original_class_1_count
    #
    #
    # class_0_mask = (y_resampled == 0)
    # original_1_mask = np.zeros_like(y_resampled, dtype=bool)
    # synthetic_1_mask = np.zeros_like(y_resampled, dtype=bool)
    #
    #
    # original_1_mask[np.where(y_resampled == 1)[0][:original_class_1_count]] = True
    # synthetic_1_mask[np.where(y_resampled == 1)[0][original_class_1_count:]] = True
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_pca[class_0_mask, 0], X_pca[class_0_mask, 1],
    #             c='tab:blue', label='Class 0 (non-toxic)', alpha=0.5, s=15)
    # plt.scatter(X_pca[original_1_mask, 0], X_pca[original_1_mask, 1],
    #             c='tab:orange', label='Class 1 (toxic, real)', alpha=0.6, s=15)
    # plt.scatter(X_pca[synthetic_1_mask, 0], X_pca[synthetic_1_mask, 1],
    #             c='tab:red', label='Class 1 (toxic, synthetic)', alpha=0.4, s=15, marker='x')
    #
    # plt.title('PCA Projection of SMOTE-Balanced Dataset')
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    #
    #
    # save_dir = 'plots'
    # save_path = os.path.join(save_dir, '0511_class_dist1.png')
    # plt.savefig(save_path)
    # print(f"Confusion matrix saved to {save_path}")
    #
    # sys.exit(0)


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
    train_set1 = MolTox_Dataset(args.train_data)
    train_loader1 = DataLoader(
        train_set1,
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

    if args.eval_only_train:
        print("Loading trained model for evaluation...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Evaluate on test data
        train_accuracy, train_loss = eval_step(
            model=model,
            device=device,
            loader=train_loader1,
            batch_size=config['train']['batch_size'],
            num_points=config['model']['max_atom_num']
        )

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Loss: {train_loss:.4f}")
        sys.exit()

    if args.eval_only:
        print("Loading trained model for evaluation...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Evaluate on test data
        metrics = evaluate_model_metrics(
            model=model,
            device=device,
            loader=valid_loader,
            num_points=config['model']['max_atom_num']
        )
        print(f"Checkpoint loaded from {args.checkpoint_path}")
        print(f"Best validation loss recorded in checkpoint: {checkpoint['best_val_mae']:.4f}")

        print(f"Test Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Test Loss: {metrics['Loss']:.4f}")
        sys.exit()

    if args.eval_only_metrics:
        print("Loading trained model for evaluation...")

        # Load model config
        with open(args.model_config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Set device
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(f"Device: {device}")

        # Load model
        model = MolnetTox_bin(config['model']).to(device)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load test set
        valid_set = MolTox_Dataset(args.test_data)
        valid_loader = DataLoader(
            valid_set,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['train']['num_workers'],
            drop_last=False
        )

        # Infer num_points from a sample
        sample_batch = next(iter(valid_loader))
        _, x_sample, _, _ = sample_batch
        num_points = x_sample.shape[1]

        # Run evaluation
        metrics, _, _ = evaluate_model_metrics(
            model=model,
            device=device,
            loader=valid_loader,
            num_points=num_points,
            return_preds_targets=True
        )

        print(f"Checkpoint loaded from {args.checkpoint_path}")
        if 'best_val_mae' in checkpoint:
            print(f"Best validation loss recorded in checkpoint: {checkpoint['best_val_mae']:.4f}")

        print(f"\nFinal Evaluation:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        if args.plot_confusion_matrix:
            cm = confusion_matrix(all_targets, all_preds)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            save_dir = 'plots/confusion_matrix'
            save_path = os.path.join(save_dir, '0511'
                                               '_mitodys_confusion_matrix.png')
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")


        sys.exit()

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=10)

    # 4. Train
    if args.resume_path != '':
        if args.transfer:
            print("Load the pretrained encoder (freeze the encoder)...")
            checkpoint = torch.load(args.resume_path, map_location=device)['model_state_dict']
            encoder_dict = {}
            for name, param in checkpoint.items():
                if not name.startswith("decoder") and not name.startswith("classifier"):
                    param.requires_grad = False #encoder won't be freezed
                    encoder_dict[name] = param
            model.load_state_dict(encoder_dict, strict=False)

        else:
            print("Load the checkpoints...")
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_valid_loss = checkpoint['best_val_loss']

    if args.checkpoint_path != '':
        checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    early_stop_step = 30
    early_stop_patience = 0

    train_losses = []
    valid_losses = []
    best_valid_loss = None

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
        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if args.checkpoint_path != '':
                print('Saving checkpoint...')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_mae': best_valid_loss,
                    'num_params': num_params
                }
                torch.save(checkpoint, args.checkpoint_path)
                print("Loss improved - checkpoint saved")
            early_stop_patience = 0 
        else:
            early_stop_patience += 1
            print(
                f"Validation loss did not improve "
                f"(current: {valid_loss:.4f}, best: {best_valid_loss:.4f}) — "
                f"{early_stop_patience}/{early_stop_step} early stop patience used"
            )

        scheduler.step(valid_loss)

        if early_stop_patience >= early_stop_step:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        print(f'Best loss so far: {best_valid_loss}')

    x = list(range(1, len(train_losses) + 1))
    print(f"Epochs: {len(x)}, Train Losses: {len(train_losses)}, Valid Losses: {len(valid_losses)}")

    # Find epoch with best validation loss
    best_epoch = valid_losses.index(min(valid_losses)) + 1

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(x, train_losses, linewidth=2.0, label="Training Loss", color="blue")
    ax.plot(x, valid_losses, linewidth=2.0, label="Validation Loss", color="red")
    ax.axvline(x=best_epoch, linestyle='--', linewidth=1.5, label=f"Best Epoch: {best_epoch}", color='green')

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend(loc="upper left")

    plot_path = args.plot
    plot_dir = os.path.dirname(plot_path)
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(plot_path)
    print(f"Plot saved at {plot_path}")



    if args.ex_model_path != '':
        print('Export the model...')
        model_scripted = torch.jit.script(model)
        model_scripted.save(args.ex_model_path)
        print(f"Model exported to {args.ex_model_path}")
