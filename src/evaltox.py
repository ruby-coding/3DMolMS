import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(args.ex_model_path, map_location=device)  # Load TorchScript model
model.to(device)
model.eval()  # Set model to evaluation mode
print("Model successfully loaded!")

def evaluate_loaded_model(model, device, test_loader, num_classes=3):
    y_true = []
    y_pred = []
    y_scores = []  # Store raw logits for AUROC calculation

    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        for batch in test_loader:
            _, x, features, y = batch  # Extract input & labels
            x = x.to(device, dtype=torch.float)
            x = x.permute(0, 2, 1)  # Ensure correct input shape
            y = y.to(device, dtype=torch.int)  # Convert labels to integers

            idx_base = torch.arange(0, x.shape[0], device=device).view(-1, 1, 1) * config['model']['max_atom_num']

            # Get model predictions
            pred_logits = model(x, None, idx_base)  # Get raw logits
            pred_probs = torch.softmax(pred_logits, dim=1)  # Convert to probability scores
            pred_classes = torch.argmax(pred_probs, dim=1)  # Get class predictions

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred_classes.cpu().numpy())
            y_scores.extend(pred_probs.cpu().numpy())  # Save probability scores for AUROC

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Compute precision, recall, F1-score
    report = classification_report(y_true, y_pred, digits=4)
    print("\nClassification Report:\n", report)

    # Compute AUROC
    try:
        y_true_one_hot = np.eye(num_classes)[y_true]  # Convert y_true to one-hot encoding
        auc_score = roc_auc_score(y_true_one_hot, y_scores, multi_class="ovr")
        print(f"\nAUROC Score: {auc_score:.4f}")
    except ValueError:
        print("\nAUROC computation failed (probably due to missing classes in the dataset).")

    return conf_matrix, auc_score

# Run evaluation on test set
conf_matrix, auc_score = evaluate_loaded_model(model, device, test_loader)
