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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

from molnetpack import MoLlamaToxClassifier
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


def train_step(model, device, loader, optimizer, batch_size) -> tuple[float, float]:
    total_loss = 0
    total_accuracy = 0
    criterion = nn.BCEWithLogitsLoss()

    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            # print(f"Batch structure: {len(batch)}")
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            # print(x.size())
            y = y.to(device, dtype=torch.float).view(-1, 1)

            optimizer.zero_grad()
            model.train()
            pred = model(x)



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



def evaluate_model_metrics(model, device, loader: DataLoader, return_preds_targets=False):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device).float()                 # Shape: [batch, 2048]
            y = y.to(device).float().view(-1)        # Shape: [batch]

            outputs = model(x).squeeze()             # logits, Shape: [batch]

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



def eval_step(model: nn.Module, device, loader: DataLoader, batch_size) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for step, batch in enumerate(loader):
                # Assuming batch = (x, y)
                _, x, _, y = batch
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float).view(-1, 1)

                logits = model(x)
                loss = criterion(logits, y)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct += (preds == y).float().sum().item()
                total += y.size(0)
                val_loss += loss.item()

                bar.set_description('Eval')
                bar.set_postfix(loss=loss.item(), acc=correct / max(1, total))
                bar.update(1)

    avg_acc = correct / max(1, total)
    avg_loss = val_loss / max(1, step + 1)
    return avg_acc, avg_loss


def init_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
    parser.add_argument('--checkpoint_path', type=str, default='./check_point/(0629)mito_mollama.pt',
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
    parser.add_argument('--plot_confusion_matrix', type=str, default='./plots/confusion_matrix',
                        help='Path to save the confusion matrix plot')
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
    train_set = MolTox_Dataset(args.train_data)

    X, y = [], []
    for i in range(len(train_set)):
        _, x_i, _, y_i = train_set[i]
        X.append(x_i.numpy() if isinstance(x_i, torch.Tensor) else x_i)
        y.append(y_i.item() if isinstance(y_i, torch.Tensor) else y_i)

    X = np.stack(X)
    y = np.array(y)

    smote = SMOTE(random_state = args.seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_tensor = torch.tensor(X_resampled, dtype=torch.float32)
    y_tensor = torch.tensor(y_resampled, dtype=torch.float32).view(-1, 1)


    data_set_smote = TensorDataset(X_tensor, y_tensor)

    train_loader = DataLoader(
        data_set_smote,
        batch_size=global_config['train']['batch_size'],
        shuffle=True,
        num_workers=global_config['train']['num_workers'],
        drop_last=True)
    train_set1 = MolTox_Dataset(args.train_data)
    train_loader1 = DataLoader(
        train_set1,
        batch_size=global_config['train']['batch_size'],
        shuffle=True,
        num_workers=global_config['train']['num_workers'],
        drop_last=True)
    valid_set = MolTox_Dataset(args.test_data)
    valid_loader = DataLoader(
        valid_set,
        batch_size=global_config['train']['batch_size'],
        shuffle=True,
        num_workers=global_config['train']['num_workers'],
        drop_last=True)

    # 2. Model
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
    print(f'Device: {device}')

    model = MoLlamaToxClassifier(global_config['model']).to(device)
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
            batch_size=global_config['train']['batch_size'],
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
            global_config = yaml.load(f, Loader=yaml.FullLoader)

        model_config = global_config['model']
        # Set device
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(f"Device: {device}")

        # Load model
        model = MoLlamaToxClassifier(model_config).to(device)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load test set
        valid_set = MolTox_Dataset(args.test_data)
        valid_loader = DataLoader(
            valid_set,
            batch_size=global_config['train']['batch_size'],
            shuffle=False,
            num_workers=global_config['train']['num_workers'],
            drop_last=False
        )



        # Run evaluation
        metrics, _, _ = evaluate_model_metrics(
            model=model,
            device=device,
            loader=valid_loader,
            return_preds_targets=True
        )

        print(f"Checkpoint loaded from {args.checkpoint_path}")
        if 'best_val_mae' in checkpoint:
            print(f"Best validation loss recorded in checkpoint: {checkpoint['best_val_mae']:.4f}")
        else:
            print("No best_val_mae found in checkpoint.")

        print(f"Evaluation Metrics:")
        for name, val in metrics.items():
            print(f"{name}: {val:.4f}")

        if args.plot_confusion_matrix != '':
            cm = confusion_matrix(all_targets, all_preds)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()

            os.makedirs(os.path.dirname(args.plot_confusion_matrix), exist_ok=True)
            plt.savefig(args.plot_confusion_matrix)
            print(f"Confusion matrix saved to {args.plot_confusion_matrix}")

        sys.exit()

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=global_config['train']['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=10)

    # 4. Train
    # 4. Resume or Transfer Learning
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

    if args.checkpoint_path != '':
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    early_stop_step = 30
    early_stop_patience = 0

    train_losses = []
    valid_losses = []
    if 'best_valid_loss' not in locals():
        best_valid_loss = None

    for epoch in range(1, global_config['train']['epochs'] + 1):
        print("\n=====Epoch {}".format(epoch))
        train_accuracy, train_loss = train_step(model, device, train_loader, optimizer,
                                                batch_size=global_config['train']['batch_size'])

        valid_accuracy, valid_loss = eval_step(model, device, valid_loader,
                                               batch_size=global_config['train']['batch_size'])


        print(
            f"Train: Accuracy: {train_accuracy}, Loss: {train_loss} \nValidation: Accuracy: {valid_accuracy}, Loss: {valid_loss}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Update best accuracy
        if best_valid_loss is None or valid_loss < best_valid_loss:
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
