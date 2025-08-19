'''
Date: 2023-10-03 21:09:14
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:16:17
'''
import os

from torch import Tensor

os.environ["SCIPY_ARRAY_API"] = "1"
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import yaml
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

from molnetpack import MolNet_LLM
from molnetpack import MolTox_LLMDataset

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


def train_step(model, device, loader, optimizer) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()
    with tqdm(total=len(loader), desc='Training') as bar:
        for batch in loader:
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float).view(-1, 1)
            optimizer.zero_grad()
            pred = model(x)

            # Loss and accuracy
            batch_loss = criterion(pred, y)
            pred_class = (pred >= 0).float()

            batch_correct = (pred_class == y).float().sum().item()
            batch_size = y.size(0)
            batch_accuracy = batch_correct / batch_size

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_correct += batch_correct
            total_samples += batch_size

            bar.set_postfix(loss=batch_loss.item(), acc=batch_accuracy, lr=get_lr(optimizer))
            bar.update(1)

    avg_loss = total_loss / max(1, len(loader))
    avg_accuracy = total_correct / max(1, total_samples)
    return avg_accuracy, avg_loss


def evaluate_model_metrics(model, device, loader: DataLoader, return_preds_targets=False):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            titles, x, y = batch
            x = x.to(device).float()
            y = y.to(device).float().view(-1)
            outputs = model(x).squeeze()
            batch_loss = criterion(outputs, y)
            total_loss += batch_loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Option 1: Manual calculation
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    # Option 2: Using sklearn (uncomment if you prefer this)
    # from sklearn.metrics import balanced_accuracy_score
    # balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy,  # Add this line
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    if return_preds_targets:
        return metrics, all_preds, all_targets
    else:
        return metrics

def eval_step(model: nn.Module, device, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for batch in loader:
                if len(batch) == 3:
                    _, x, y = batch
                else:
                    x, y = batch
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float).view(-1, 1)
                logits = model(x)
                batch_loss = criterion(logits, y)

                preds = (logits >= 0).float()

                batch_correct = (preds == y).float().sum().item()
                batch_size = y.size(0)

                total_correct += batch_correct
                total_samples += batch_size
                total_loss += batch_loss.item()

                bar.update(1)
                bar.set_postfix(loss=batch_loss.item(), acc=total_correct / max(1, total_samples))

    avg_acc = total_correct / max(1, total_samples)
    avg_loss = total_loss / max(1, len(loader))
    return avg_acc, avg_loss

def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def apply_smote_to_train_set(train_set: MolTox_LLMDataset) -> TensorDataset:
    X, y = [], []
    for i in range(len(train_set)):
        _, mol_embedding, label = train_set[i]
        X.append(mol_embedding.numpy() if isinstance(mol_embedding, torch.Tensor) else mol_embedding)
        y.append(label.item() if isinstance(label, torch.Tensor) else label[0])  # Handle numpy array label
    X = np.stack(X)
    y = np.array(y)
    smote = SMOTE(random_state=args.seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_tensor = torch.tensor(X_resampled, dtype=torch.float32)
    y_tensor = torch.tensor(y_resampled, dtype=torch.float32).view(-1, 1)
    return TensorDataset(X_tensor, y_tensor)


def only_evaluate_on_train_set(checkpoint_path, device, model, loader: DataLoader[tuple[Tensor,...]]):
    print("Loading trained model for evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Evaluate on test data
    train_accuracy, train_loss = eval_step(
        model=model,
        device=device,
        loader=loader
    )
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Loss: {train_loss:.4f}")
    sys.exit()


def only_evaluate_on_val(checkpoint_path, device, model, valid_loader):
    print("Loading trained model for evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # Evaluate on test data
    metrics = evaluate_model_metrics(
        model=model,
        device=device,
        loader=valid_loader,
    )
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Best validation loss recorded in checkpoint: {checkpoint['best_val_loss']:.4f}")
    print(f"Test Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Test Loss: {metrics['Loss']:.4f}")
    sys.exit()

def plot_confusion_matrix_func(all_targets, all_preds, save_path):
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


def only_evaluation_metrics(model_config_path, checkpoint_path, test_data_path, device, no_cuda, batch_size,
                            num_workers, return_preds=False):
    print("Loading trained model for evaluation...")

    # Load model config
    with open(model_config_path, 'r') as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)
    model_config = global_config['model']

    # Set device
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and not no_cuda else "cpu")
    print(f"Device: {device}")

    # Load model
    model = MolNet_LLM(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test set
    valid_set = MolTox_LLMDataset(test_data_path)
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    # Run evaluation
    if return_preds:
        # This would require modifying evaluate_model_metrics to return predictions
        metrics, all_preds, all_targets = evaluate_model_metrics(
            model=model,
            device=device,
            loader=valid_loader,
            return_preds_targets=True
        )
    else:
        metrics = evaluate_model_metrics(
            model=model,
            device=device,
            loader=valid_loader,
            return_preds_targets=False
        )

    print(f"Checkpoint loaded from {checkpoint_path}")
    if 'best_val_mae' in checkpoint:
        print(f"Best validation loss recorded in checkpoint: {checkpoint['best_val_mae']:.4f}")
    else:
        print("No best_val_mae found in checkpoint.")

    print(f"Evaluation Metrics:")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    if return_preds:
        return metrics, all_preds, all_targets
    else:
        return metrics

def transfer_learning(args, model, optimizer, scheduler, device):
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path, map_location=device)

        if args.transfer:
            print("Load the pretrained encoder (freeze the decoder)")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Freeze encoder layers (actual model parameters)
            for name, param in model.named_parameters():
                if not name.startswith("decoder") and not name.startswith("classifier"):
                    param.requires_grad = False

        else:
            print("Load the full model checkpoint...")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Try multiple keys in case your checkpoints used different keys
            if 'best_val_loss' in checkpoint:
                best_valid_loss = checkpoint['best_val_loss']
            elif 'best_val_mae' in checkpoint:
                best_valid_loss = checkpoint['best_val_mae']
            else:
                print("Warning: No best_val_loss or best_val_mae in checkpoint")
                best_valid_loss = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Toxicity Prediction Model using MolLama')
    parser.add_argument('--train_data', type=str, default='./data/mito_mollama_train.pkl',
                        help='path to training data (pkl)')
    parser.add_argument('--test_data', type=str, default='./data/mito_mollama_test.pkl',
                        help='path to test data (pkl)')
    parser.add_argument('--model_config_path', type=str, default='./src/molnetpack/config/molnet_llm.yml',
                        help='path to model and training configuration')
    parser.add_argument('--data_config_path', type=str, default='./src/molnetpack/config/preprocess_etkdgv3.yml',
                        help='path to configuration')
    parser.add_argument('--checkpoint_path', type=str, default='./check_point/(0731)mito_mollama.pt',
                        help='Path to save checkpoint') #Change
    parser.add_argument('--resume_path', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--transfer', action='store_true',
                        help='Whether to load the pretrained encoder')
    parser.add_argument('--ex_model_path', type=str, default='',
                        help='Path to export the whole model (structure & weights)')
    parser.add_argument('--validation_only', action='store_true',
                        help='Run validation only without training')
    parser.add_argument('--plot', type=str, default='./plots/0731_mito_mollama',
                        help='Directory to save the plot') #Change
    parser.add_argument('--plot_confusion_matrix', type=str, default='./plots/confusion_matrix/0731_mito_mollama',
                        help='Path to save the confusion matrix plot') #Change
    parser.add_argument('--eval_only', action='store_true', help="Only evaluate the model without training")
    parser.add_argument('--eval_only_train', action='store_true', help="Only evaluate the model without training")
    parser.add_argument('--eval_only_metrics', action='store_true', help="Only evaluate the model with metrics without training")


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
        global_config = yaml.load(f, Loader=yaml.FullLoader)

    print(f'Load the model & training configuration from {args.model_config_path}')


    # 1. Data
    train_set = MolTox_LLMDataset(args.train_data)

    train_set_SMOTE: TensorDataset = apply_smote_to_train_set(train_set)
    del train_set

    train_loader = DataLoader(
        train_set_SMOTE,
        batch_size=global_config['train']['batch_size'],
        shuffle=True,
        num_workers=global_config['train']['num_workers'],
        drop_last=True)

    valid_set = MolTox_LLMDataset(args.test_data)
    valid_loader = DataLoader(
        valid_set,
        batch_size=global_config['train']['batch_size'],
        shuffle=False,
        num_workers=global_config['train']['num_workers'],
        drop_last=True)

    # 2. Model
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    print(f'Device: {device}')

    model = MolNet_LLM(global_config['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{str(model)} #Params: {num_params}')

    if args.eval_only_train:
        only_evaluate_on_train_set(args.checkpoint_path, device, model, train_loader)

    if args.eval_only:
        only_evaluate_on_val(args.checkpoint_path, device, model, valid_loader)

    if args.eval_only_metrics:
        # Option 1: If you only want metrics and no confusion matrix
        only_evaluation_metrics(
            model_config_path=args.model_config_path,
            checkpoint_path=args.checkpoint_path,
            test_data_path=args.test_data,
            device=args.device,  # Pass just the device number, not formatted string
            no_cuda=args.no_cuda,
            batch_size=global_config['train']['batch_size'],
            num_workers=global_config['train']['num_workers'],
        )

        # Option 2: If you need confusion matrix, you need to modify the function call
        if args.plot_confusion_matrix != '':
            # You'll need to modify only_evaluation_metrics to also return predictions
            # OR call evaluate_model_metrics directly here to get predictions
            print(
                "Warning: Confusion matrix plotting requires predictions, but only_evaluation_metrics doesn't return them")

        sys.exit()

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=global_config['train']['lr'],
                            weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.3, patience=5)

    # 4. Train
    # 4. Resume or Transfer Learning
    best_valid_loss = transfer_learning(args, model, optimizer, scheduler, device)

    if args.checkpoint_path != '':
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    early_stop_step = 50
    early_stop_patience = 0

    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')

    for epoch in range(1, global_config['train']['epochs'] + 1):
        print("\n=====Epoch {}".format(epoch))
        train_accuracy, train_loss = train_step(model, device, train_loader, optimizer)

        valid_accuracy, valid_loss = eval_step(model, device, valid_loader)


        print(
            f"Train: Accuracy: {train_accuracy}, Loss: {train_loss} \nValidation: Accuracy: {valid_accuracy}, Loss: {valid_loss}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Update best accuracy
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_patience = 0
            if args.checkpoint_path != '':
                print('Saving checkpoint...')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_valid_loss,
                    'num_params': num_params
                }
                torch.save(checkpoint, args.checkpoint_path)
                print("Loss improved - checkpoint saved")

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

    # Best epoch (1-based)
    best_epoch = valid_losses.index(min(valid_losses)) + 1

    fig, ax = plt.subplots()
    ax.plot(x, train_losses, label="Training Loss", color="blue", linewidth=2.0)
    ax.plot(x, valid_losses, label="Validation Loss", color="red", linewidth=2.0)
    ax.axvline(x=best_epoch, linestyle='--', color='green', linewidth=1.5, label=f"Best Epoch: {best_epoch}")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()

    # Save
    plot_path = args.plot
    plot_dir = os.path.dirname(plot_path)
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(plot_path)
    print(f"📈 Plot saved at {plot_path}")

    if args.ex_model_path != '':
        print('Export the model...')
        model_scripted = torch.jit.script(model)
        model_scripted.save(args.ex_model_path)
        print(f"Model exported to {args.ex_model_path}")
