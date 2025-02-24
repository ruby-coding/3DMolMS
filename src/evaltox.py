import torch
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from torch.utils.data import DataLoader
from molnetpack import MolTox_Dataset  # Ensure this is correctly imported

# Argument parser for loading saved model
parser = argparse.ArgumentParser(description="Evaluate trained model")
parser.add_argument('--test_data', type=str, default='./data/cardio_toxicity_etkdgv3_test.pkl',
                    help='Path to test data (pkl)')
parser.add_argument('--ex_model_path', type=str, required=True,
                    help='Path to exported model (TorchScript)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for evaluation')
parser.add_argument('--device', type=int, default=0,
                    help='Which GPU to use if any')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()

# Set device
device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() and not args.no_cuda else "cpu")
print(f"Using device: {device}")

# Load the trained model
print("Loading trained model...")
model = torch.jit.load(args.ex_model_path, map_location=device)
model.to(device)
model.eval()
print("Model successfully loaded!")
print("Model")
# Load test data
test_set = MolTox_Dataset(args.test_data)
test_loader = DataLoader(test_set, batch_size=config['train']['batch_size'], shuffle=False)

# Retrieve num_points from config (as done during training)
num_points = config['model']['max_atom_num']

def evaluate_model(model, device, test_loader, num_points, num_classes=3):
    y_true = []
    y_pred = []
    y_scores = []  # Store logits for AUROC calculation

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            _, x, features, y = batch  # Extract input & labels
            x = x.to(device, dtype=torch.float)
            x = x.permute(0, 2, 1)  # Ensure correct shape
            y = y.to(device, dtype=torch.int)  # Convert labels to integers

            # Correct the idx_base calculation using explicit num_points
            idx_base = torch.arange(0, x.shape[0], device=device).view(-1, 1, 1) * num_points

            # Get predictions
            pred_logits = model(x, None, idx_base)  # Get raw logits
            pred_probs = torch.softmax(pred_logits, dim=1)  # Convert logits to probabilities
            pred_classes = torch.argmax(pred_probs, dim=1)  # Convert probabilities to class predictions

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred_classes.cpu().numpy())
            y_scores.extend(pred_probs.cpu().numpy())  # Save probability scores for AUROC

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Compute precision, recall, F1-score
    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report:\n", report)

    # Compute AUROC (if applicable)
    auc_score = None
    try:
        y_true_one_hot = np.eye(num_classes)[y_true]  # Convert labels to one-hot
        auc_score = roc_auc_score(y_true_one_hot, y_scores, multi_class="ovr")
        print(f"\nAUROC Score: {auc_score:.4f}")
    except ValueError:
        print("\nAUROC computation failed (likely due to missing classes in dataset).")

    return conf_matrix, auc_score

# Run evaluation
conf_matrix, auc_score = evaluate_model(model, device, test_loader, num_points)
