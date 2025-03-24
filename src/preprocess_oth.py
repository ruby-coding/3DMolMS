import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd 

from rdkit import Chem
# ignore the warning
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from molnetpack import conformation_array, filter_mol, check_atom
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


def check_valid_smiles(smiles):
	"""Check if the SMILES string represents a valid molecule."""
	mol = Chem.MolFromSmiles(smiles)
	return mol is not None


def random_split(suppl, smiles_list, test_ratio=0.1):
	test_smiles = np.random.choice(smiles_list, int(len(smiles_list)*test_ratio), replace=False)

	train_mol = []
	test_mol = []
	for mol in suppl: 
		smiles = Chem.MolToSmiles(mol)
		if smiles in test_smiles:
			test_mol.append(mol)
		else:
			train_mol.append(mol)
	return test_mol, train_mol

def sdf2arr(suppl, encoder):
	'''data format
	[
		{'title': <str>, 'mol': <numpy array>, 'rt': <numpy array>}, 
		{'title': <str>, 'mol': <numpy array>, 'rt': <numpy array>}, 
		....
	]
	'''
	data = []
	for idx, mol in enumerate(tqdm(suppl)): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=Chem.MolToSmiles(mol), 
															conf_type=encoder['conf_type']) 
		# There are some limitations of conformation generation methods. 
		# e.g. https://github.com/rdkit/rdkit/issues/5145
		# Let's skip the unsolvable molecules. 
		if not good_conf: 
			continue
		atom_type_one_hot = np.array([encoder['atom_type'][atom] for atom in atom_type])
		assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
		mol_arr = np.pad(mol_arr, ((0, encoder['max_atom_num']-xyz_arr.shape[0]), (0, 0)), constant_values=0)
		
		data.append({'title': mol.GetProp('PUBCHEM_COMPOUND_CID'), 'mol': mol_arr, 'rt': np.array([mol.GetProp('RETENTION_TIME')]).astype(np.float64)})
	return data

def df2arr(df, encoder): 
	'''data format
	[
		{'title': <str>, 'mol': <numpy array>, 'rt': <numpy array>}, 
		{'title': <str>, 'mol': <numpy array>, 'rt': <numpy array>}, 
		....
	]
	'''
	data = []
	for idx, row in tqdm(df.iterrows(), total=df.shape[0]): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=row['Structure'], 
															conf_type=encoder['conf_type']) 
		# There are some limitations of conformation generation methods. 
		# e.g. https://github.com/rdkit/rdkit/issues/5145
		# Let's skip the unsolvable molecules. 
		if not good_conf: 
			continue
		atom_type_one_hot = np.array([encoder['atom_type'][atom] for atom in atom_type])
		assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
		mol_arr = np.pad(mol_arr, ((0, encoder['max_atom_num']-xyz_arr.shape[0]), (0, 0)), constant_values=0)
		
		env_arr = np.array(encoder['precursor_type'][row['Adduct']])

		data.append({'title': row['AllCCS ID']+'_'+str(idx), 'mol': mol_arr, 'ccs': np.array([row['CCS']]).astype(np.float64), 'env': env_arr})
	return data

def csv2arr(df, encoder):
	'''
	    Converts a DataFrame containing SMILES, labels, and molecular descriptors into structured NumPy arrays.

	    Output format:
	    [
	        {'title': <str>, 'mol': <numpy array>, 'features': <numpy array>, 'label': <int>},
	        {'title': <str>, 'mol': <numpy array>, 'features': <numpy array>, 'label': <int>},
	        ...
	    ]
	'''
	data = []

	for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
		# Extract SMILES and label
		smiles = row['smiles']
		label = int(row['labels'])

		# Generate molecular conformation
		good_conf, xyz_arr, atom_type = conformation_array(smiles=smiles, conf_type=encoder['conf_type'])

		# Skip molecules that couldn't generate valid conformations
		if not good_conf:
			continue

		# Convert atom types to one-hot encoding
		atom_type_one_hot = np.array([encoder['atom_type'][atom] for atom in atom_type])
		assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

		# Merge atomic coordinates with atom encoding
		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)

		# Pad molecules to a fixed number of atoms
		mol_arr = np.pad(mol_arr, ((0, encoder['max_atom_num'] - xyz_arr.shape[0]), (0, 0)), constant_values=0)

		# Extract numerical feature columns
		feature_values = row.iloc[2:].values.astype(np.float64)  # Exclude 'labels' and 'smiles' columns

		# Store in the structured format
		data.append({
			'title': f"Mol_{idx}",  # Unique molecule ID
			'mol': mol_arr,  # Molecular representation
			'features': feature_values,  # Molecular descriptors/features
			'label': label  # Classification label
		})

	return data



	


if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Preprocess the Data')
	parser.add_argument('--raw_dir', type=str, default='./data/origin/',
						help='path to raw data')
	parser.add_argument('--pkl_dir', type=str, default='./data/',
						help='path to pkl data')
	parser.add_argument('--dataset', type=str, nargs='+', required=True, choices=['metlin', 'allccs', 'cardio_toxicity','increase_mitochondrial_dysfunction'],
						help='dataset name')
	parser.add_argument('--data_config_path', type=str, default='./src/molnetpack/config/preprocess_etkdgv3.yml',
						help='path to configuration')
	args = parser.parse_args()
	
	if 'metlin' in args.dataset: 
		assert os.path.exists(os.path.join(args.raw_dir, 'SMRT_dataset.sdf'))
	if 'allccs' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'allccs_download.csv'))
	if 'cardio_toxicity' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'cardio_toxicity.csv'))
	if 'increase_mitochondrial_dysfunction' in args.dataset:
		assert os.path.exists(os.path.join(args.raw_dir, 'increase_mitochondrial_dysfunction.csv'))
	
	# load the configurations
	with open(args.data_config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)
	
	if 'metlin' in args.dataset: 
		# 1. load data
		print('\n>>> Step 1: load the dataset;')
		suppl = Chem.SDMolSupplier(os.path.join(args.raw_dir, 'SMRT_dataset.sdf'))
		suppl = [m for m in suppl if m != None and m.HasProp('PUBCHEM_COMPOUND_CID') and m.HasProp('RETENTION_TIME')]
		print('Load {} data from METLIN Dataset...'.format(len(suppl)))

		# 2. randomly split spectra into training and test set according to [smiles]
		print('\n>>> Step 2: filter out molecules by certain rules; randomly split SMILES into training set and test set;')
		suppl, smiles_list = filter_mol(suppl, config['metlin_rt'])
		test_mol, train_mol = random_split(suppl, 
									 		list(set(smiles_list)), 
											test_ratio=0.1)
		print('Get {} test data and {} training data'.format(len(test_mol), len(train_mol)))

		# 3. encoding data into arrays
		print('\n>>> Step 3: encode all the data into pkl format;')
		test_data = sdf2arr(test_mol, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'metlin_{}_test.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))
			
		train_data = sdf2arr(train_mol, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'metlin_{}_train.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))

	if 'allccs' in args.dataset:
		# 1. load data
		print('\n>>> Step 1: load the dataset;')
		df = pd.read_csv(os.path.join(args.raw_dir, 'allccs_download.csv'), on_bad_lines='skip')
		df = df[df['Type'] == 'Experimental CCS']
		df = df.dropna(subset=['Structure', 'Adduct', 'CCS'])
		# df.to_csv(os.path.join(args.raw_dir, 'allccs_experiment.csv')) # tmp
		# df = pd.read_csv(os.path.join(args.raw_dir, 'allccs_experiment.csv'), on_bad_lines='skip') # tmp
		print('Load {} data from AllCCS Dataset...'.format(len(df)))

		# 2. randomly split spectra into training and test set according to [smiles]
		print('\n>>> Step 2: filter out molecules by certain rules; randomly split SMILES into training set and test set;')
		df['atom_check'] = df['Structure'].apply(lambda x: check_atom(x, config['allccs'], in_type='smiles'))
		df = df[df['atom_check'] == True]
		df = df[df['Adduct'].isin(config['allccs']['precursor_type'])]
		
		smiles_list = list(set(df['Structure'].tolist()))
		test_smiles = np.random.choice(smiles_list, int(len(smiles_list)*0.1), replace=False)
		train_smiles = [s for s in smiles_list if s not in test_smiles]
		test_df = df[df['Structure'].isin(test_smiles)]
		train_df = df[df['Structure'].isin(train_smiles)]
		print('Get {} test data and {} training data'.format(len(test_df), len(train_df)))

		# 3. encoding data into arrays
		print('\n>>> Step 3: encode all the data into pkl format;')
		test_data = df2arr(test_df, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'allccs_{}_test.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))
			
		train_data = df2arr(train_df, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'allccs_{}_train.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f: 
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))

	if 'cardio_toxicity' in args.dataset:
		print('\n>>> Step 1: load the dataset;')
		df = pd.read_csv(os.path.join(args.raw_dir, 'cardio_toxicity.csv'))
		df = df.dropna(subset=['smiles', 'labels'])
		print('Load {} data from Cardio-Toxicity Dataset...'.format(len(df)))

		print('\n>>> Step 2: filter out invalid molecules; randomly split SMILES into training and test sets;')
		df['valid'] = df['smiles'].apply(lambda x: check_atom(x, config['cardio_toxicity'], in_type='smiles')) #filter out the compounds
		df = df[df['valid'] == True]

		test_ratio = 0.1
		df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset
		test_size = int(len(df) * test_ratio)
		test_df = df.iloc[:test_size]
		train_df = df.iloc[test_size:]
		print('Get {} test data and {} training data'.format(len(test_df), len(train_df)))

		print('\n>>> Step 3: encode all the data into pkl format;')
		test_data = csv2arr(test_df, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'cardio_toxicity_{}_test.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f:
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))

		train_data = csv2arr(train_df, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'cardio_toxicity_{}_train.pkl'.format(config['encoding']['conf_type']))
		with open(out_path, 'wb') as f:
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))

	if 'increase_mitochondrial_dysfunction' in args.dataset:
		print('\n>>> Step 1: load the dataset;')
		df = pd.read_csv(os.path.join(args.raw_dir, 'increase_mitochondrial_dysfunction.csv'))
		df = df.dropna(subset=['smiles', 'labels'])
		print('Load {} data from Increase Mitochondrial Dysfunction Dataset...'.format(len(df)))

		print('\n>>> Step 2: filter out invalid molecules; randomly split SMILES into training and test sets;')
		df['valid'] = df['smiles'].apply(
			lambda x: check_atom(x, config['increase_mitochondrial_dysfunction'], in_type='smiles'))
		df = df[df['valid'] == True]

		test_ratio = 0.2
		df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset
		test_size = int(len(df) * test_ratio)
		test_df = df.iloc[:test_size]
		train_df = df.iloc[test_size:]
		print('Get {} test data and {} training data'.format(len(test_df), len(train_df)))

		print('\n>>> Step 3: encode all the data into pkl format;')
		test_data = csv2arr(test_df, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'increase_mitochondrial_dysfunction_{}_test.pkl'.format(
			config['encoding']['conf_type']))
		with open(out_path, 'wb') as f:
			pickle.dump(test_data, f)
			print('Save {}'.format(out_path))

		train_data = csv2arr(train_df, config['encoding'])
		out_path = os.path.join(args.pkl_dir, 'increase_mitochondrial_dysfunction_{}_train.pkl'.format(
			config['encoding']['conf_type']))
		with open(out_path, 'wb') as f:
			pickle.dump(train_data, f)
			print('Save {}'.format(out_path))



