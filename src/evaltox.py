import torch
from torch.utils.data import DataLoader
import argparse
from your_model_script import MolNet_tox  # Replace with actual import
from your_data_script import MolTox_Dataset  # Replace with actual import
import yaml

# Define argument parser for loading paths
parser = argparse.ArgumentParser(description="Evaluate model accuracy from checkpoint")
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to saved checkpoint')
parser.add_argument('--test_data', type=str, required=True, help='Path to test data (pkl)')
parser.add_argument('--model_config_path', type=str, required=True, help='Path to model config YAML file')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
parser.add_argument('--device', type=int, default=0, help='GPU device ID (use -1 for CPU)')
args = parser.parse_args()

# Load configuration
with open(args.model_config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Set device
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

# Load dataset
test_set = MolTox_Dataset(args.test_data)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Load model
model = MolNet_tox(config['model']).to(device)

# Load checkpoint
print(f"Loading model from {args.checkpoint_path}...")
checkpoint = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluation function
def evaluate_model(model, device, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            _, x, features, y = batch  # Adjust this if your dataset has a different format
            x = x.to(device, dtype=torch.float)
            x = x.permute(0, 2, 1)
            y = y.to(device, dtype=torch.long)

            # Get predictions
            outputs = model(x, None, None)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy

# Run evaluation
accuracy = evaluate_model(model, device, test_loader)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")
#
# python evaluate.py --checkpoint_path ./check_point/molnet_rt_etkdgv3.pt \
#                    --test_data ./data/cardio_toxicity_etkdgv3_test.pkl \
#                    --model_config_path ./src/molnetpack/config/molnet_rt.yml \
#                    --batch_size 32 \
#                    --device 0

