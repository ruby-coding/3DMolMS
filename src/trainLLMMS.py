import os
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from molnetpack import MolNet_MS
from molnetpack import MolLLMMS_Dataset
from molnetpack import __version__


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


def reg_criterion(outputs, targets):
	# cosine similarity
	t = nn.CosineSimilarity(dim=1)
	spec_cosi = torch.mean(1 - t(outputs, targets))
	return spec_cosi


def train_step(model, device, loader, optimizer):
	def train_step(model, device, loader, optimizer):
		accuracy = 0
		model.train()  # Move model.train() outside the loop

		with tqdm(total=len(loader)) as bar:
			for step, batch in enumerate(loader):
				_, x, mask, y, env = batch
				current_batch_size = x.size(0)  # Get actual batch size
				num_points = x.size(1)  # Get actual number of points

				# Move tensors to device
				x = x.to(device=device, dtype=torch.float)
				x = x.permute(0, 2, 1)
				mask = mask.to(device=device)
				y = y.to(device=device, dtype=torch.float)

				# Convert env dict values to tensors on the correct device
				for k, v in env.items():
					env[k] = torch.tensor(v, device=device, dtype=torch.float)

				# Use actual batch size, not fixed parameter
				idx_base = torch.arange(0, current_batch_size, device=device).view(-1, 1, 1) * num_points

				optimizer.zero_grad()
				pred = model(x, mask, env, idx_base)
				loss = reg_criterion(pred, y)
				loss.backward()
				optimizer.step()

				bar.set_description('Train')
				bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
				bar.update(1)

				# Recover sqrt spectra to original spectra
				with torch.no_grad():  # Prevent gradient computation for accuracy calculation
					y_orig = torch.pow(y, 2)
					pred_orig = torch.pow(pred.detach(), 2)  # Use detached prediction
					accuracy += F.cosine_similarity(pred_orig, y_orig, dim=1).mean().item()

		return accuracy / (step + 1)

def eval_step(model, device, loader):
		model.eval()
		accuracy = 0

		with tqdm(total=len(loader)) as bar:
			for step, batch in enumerate(loader):
				_, x, mask, y, env = batch
				current_batch_size = x.size(0)  # Get actual batch size
				num_points = x.size(1)  # Get actual number of points

				# Move tensors to device
				x = x.to(device=device, dtype=torch.float)
				x = x.permute(0, 2, 1)
				mask = mask.to(device=device)
				y = y.to(device=device, dtype=torch.float)

				# Convert env dict values to tensors on the correct device
				for k, v in env.items():
					env[k] = torch.tensor(v, device=device, dtype=torch.float)

				# Use actual batch size, not fixed parameter
				idx_base = torch.arange(0, current_batch_size, device=device).view(-1, 1, 1) * num_points

				with torch.no_grad():
					pred = model(x, mask, env, idx_base)

					# Normalize per sample, not globally
					pred = pred / (
								torch.max(pred, dim=1, keepdim=True)[0] + 1e-8)  # Add epsilon to avoid division by zero

					# Recover sqrt spectra to original spectra
					y_orig = torch.pow(y, 2)
					pred_orig = torch.pow(pred, 2)

					# Post-process - apply threshold
					pred_orig = torch.where(pred_orig > 0.01, pred_orig, torch.zeros_like(pred_orig))

					accuracy += F.cosine_similarity(pred_orig, y_orig, dim=1).mean().item()

				bar.set_description('Eval')
				bar.update(1)

		return accuracy / (step + 1)


def init_random_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True  # Add for full reproducibility
	torch.backends.cudnn.benchmark = False
	return



if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Molecular Mass Spectra Prediction (Train)')
	parser.add_argument('--train_data', type=str, default='./data/msmollama_etkdgv3_train.pkl',
						help='path to training data (pkl)')
	parser.add_argument('--test_data', type=str, default='./data/msmollama_etkdgv3_test.pkl',
						help='path to test data (pkl)')
	parser.add_argument('--precursor_type', type=str, default='All', choices=['All', '[M+H]+', '[M-H]-'], 
                        help='Precursor type')
	parser.add_argument('--model_config_path', type=str, default='./src/molnetpack/config/molnet.yml',
						help='path to model and training configuration')
	parser.add_argument('--data_config_path', type=str, default='./src/molnetpack/config/preprocess_etkdgv3.yml',
						help='path to configuration')
	parser.add_argument('--checkpoint_path', type=str, default = './check_point/(0907)msmollama.pt',
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
	parser.add_argument('--no_cuda', action='store_true', 
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
	# convert precursor type to encoded precursor type for filtering
	with open(args.data_config_path, 'r') as f:
		tmp = yaml.load(f, Loader=yaml.FullLoader)
		precursor_encoder = {}
		for k, v in tmp['encoding']['precursor_type'].items():
			precursor_encoder[k] = ','.join([str(int(i)) for i in v])
		precursor_encoder['All'] = False
		del tmp

	train_set = MolLLMMS_Dataset(args.train_data, precursor_type=precursor_encoder[args.precursor_type])
	train_loader = DataLoader(
		train_set,
		batch_size=config['train']['batch_size'],
		shuffle=True,
		num_workers=config['train']['num_workers'],
		drop_last=True)
	valid_set = MolLLMMS_Dataset(args.test_data, precursor_type=precursor_encoder[args.precursor_type],
								 data_augmentation=False)  # No augmentation for validation
	valid_loader = DataLoader(
		valid_set,
		batch_size=config['train']['batch_size'],
		shuffle=False,  # Don't shuffle validation data
		num_workers=config['train']['num_workers'],
		drop_last=True)

	# 2. Model
	device = torch.device(
		"cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
	print(f'Device: {device}')

	model = MolNet_MS(config['model']).to(device)
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')

	# 3. Train
	optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

	# Initialize best_valid_acc before checkpoint loading
	best_valid_acc = 0

	if args.transfer and args.resume_path != '':
		print("Load the pretrained encoder (freeze the encoder)...")
		checkpoint = torch.load(args.resume_path, map_location=device, weights_only=True)
		state_dict = checkpoint['model_state_dict']
		encoder_dict = {}
		for name, param in state_dict.items():
			if not name.startswith("decoder"):
				param.requires_grad = False  # freeze the encoder
				encoder_dict[name] = param
		model.load_state_dict(encoder_dict, strict=False)
		# Load best validation accuracy if available
		if 'best_val_acc' in checkpoint:
			best_valid_acc = checkpoint['best_val_acc']
	elif args.resume_path != '':
		print("Load the checkpoints...")
		checkpoint = torch.load(args.resume_path, map_location=device, weights_only=True)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		best_valid_acc = checkpoint['best_val_acc']

	if args.checkpoint_path != '':
		checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
		os.makedirs(checkpoint_dir, exist_ok=True)

	early_stop_step = 10
	early_stop_patience = 0

	for epoch in range(1, config['train']['epochs'] + 1):
		print("\n=====Epoch {}".format(epoch))

		# Use corrected training functions without fixed batch_size/num_points parameters
		train_acc = train_step(model, device, train_loader, optimizer)
		valid_acc = eval_step(model, device, valid_loader)

		print("Train: Acc: {:.4f}, \nValidation: Acc: {:.4f}".format(train_acc, valid_acc))

		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc

			if args.checkpoint_path != '':
				print('Saving checkpoint...')
				checkpoint = {'version': __version__,
							  'epoch': epoch,
							  'model_state_dict': model.state_dict(),
							  'optimizer_state_dict': optimizer.state_dict(),
							  'scheduler_state_dict': scheduler.state_dict(),
							  'best_val_acc': best_valid_acc,
							  'num_params': num_params}
				torch.save(checkpoint, args.checkpoint_path)

			early_stop_patience = 0
			print('Early stop patience reset')
		else:
			early_stop_patience += 1
			print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

		# scheduler.step()
		scheduler.step(valid_acc)  # ReduceLROnPlateau
		print(f'Best cosine similarity so far: {best_valid_acc:.4f}')

		if early_stop_patience == early_stop_step:
			print('Early stop!')
			break

	if args.ex_model_path != '':  # export the model
		print('Export the model...')
		# model_scripted = torch.jit.script(model) # Export to TorchScript
		# model_scripted.save(args.ex_model_path) # Save

		print('Export the traced model...')
		batch_size = config['train']['batch_size']  # Set to the desired batch size

		# Get embedding dimension from dataset instead of hardcoded values
		embedding_dim = train_set.get_embedding_dim()

		# Create example inputs with the same data types and shapes as expected by your model
		# Note: For embeddings, the input shape should match the embedding dimension
		x = torch.randn(batch_size, embedding_dim, device=device, dtype=torch.float)
		mask = torch.ones(batch_size, embedding_dim, device=device, dtype=torch.bool)

		# Get environment dimension from actual data or config
		env_dim = config['model'].get('add_num', 6)  # Default to 6 if not specified
		env = torch.randn(batch_size, env_dim, device=device, dtype=torch.float)

		# For embeddings, idx_base might not be needed or should be adjusted
		idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * embedding_dim

		example_inputs = (x, mask, env, idx_base)

		model.eval()
		try:
			traced_model = torch.jit.trace(model, example_inputs)
			torch.jit.save(traced_model, args.ex_model_path)
			print(f'Model successfully exported to {args.ex_model_path}')
		except Exception as e:
			print(f'Error during model export: {e}')
			print('Skipping model export...')
