'''
Date: 2023-10-02 20:24:27
LastEditors: yuhhong
LastEditTime: 2023-10-20 17:01:37
'''
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



class MolMS_Dataset(Dataset):
	def __init__(self, x, data_augmentation=True, precursor_type=False, mode='path'):
		if mode == 'path':
			path = x
			with open(path, 'rb') as file:
				data = pickle.load(file)
		elif mode == 'data':
			data = x
			path = 'unknown'
		else:
			raise ValueError('Unsupported mode:', mode)

		if precursor_type:
			data = self.filter_precursor_type(data, precursor_type)

		# generate mask
		for idx in range(len(data)):
			mask = ~np.all(data[idx]['mol'] == 0, axis=1)
			data[idx]['mask'] = mask.astype(bool)

		# data augmentation by flipping the x,y,z-coordinates
		if data_augmentation:
			flipping_data = []
			for d in data:
				flipping_mol_arr = np.copy(d['mol'])
				flipping_mol_arr[:, 0] *= -1
				flipping_data.append({'title': d['title']+'_f', 'mol': flipping_mol_arr, 'spec': d['spec'], 'env': d['env']})

			self.data = data + flipping_data
			print('Load {} data (with data augmentation by flipping coordinates)'.format(len(self.data)))
		else:
			self.data = data
			print('Load {} data'.format(len(self.data)))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['mask'], self.data[idx]['spec'], self.data[idx]['env']

	def filter_precursor_type(self, data, precursor_type):
		filtered_data = []
		for d in data:
			d_precursor_type = ','.join([str(int(i)) for i in d['env'][1:]])
			if d_precursor_type == precursor_type:
				filtered_data.append(d)
		return filtered_data


class MolLLMMS_Dataset(Dataset):
    def __init__(self, x, data_augmentation=True, precursor_type=False, mode='path'):
        if mode == 'path':
            path = x
            if path.endswith('.pkl'):
                try:
                    data = pd.read_pickle(path)
                    print(f"Loaded pickle file (pandas pickle): {path}")
                except Exception:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    print(f"Loaded pickle file (list of dicts): {path}")
            elif path.endswith('.csv'):
                data = pd.read_csv(path)
                print(f"Loaded CSV file: {path}")
            else:
                raise ValueError(f"Unsupported file format for {path}")
        elif mode == 'data':
            data = x
            path = 'unknown'
        else:
            raise ValueError('Unsupported mode:', mode)

        # ---------- Branch: DataFrame ----------
        if isinstance(data, pd.DataFrame):
            print(f"Data shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")

            # Extract embeddings
            embedding_cols = [col for col in data.columns if col.startswith('molembed')]
            if not embedding_cols:
                raise ValueError("No embedding columns found in DataFrame")

            self.embeddings = data[embedding_cols].values.astype(np.float32)
            self.mol_ids = data['mol_id'].values if 'mol_id' in data.columns else np.arange(len(data))
            self.smiles = data['smiles'].values if 'smiles' in data.columns else ['unknown'] * len(data)

            # Store other metadata
            self.metadata = {}
            metadata_cols = [c for c in data.columns if c not in embedding_cols + ['mol_id', 'smiles']]
            for col in metadata_cols:
                self.metadata[col] = data[col].values

        # ---------- Branch: List of Dicts ----------
        elif isinstance(data, list) and isinstance(data[0], dict):
            print(f"Data loaded as list of {len(data)} dicts")
            # Assume keys: 'title', 'mol', 'mol_meta'
            self.embeddings = np.stack([d['mol'] for d in data]).astype(np.float32)
            self.mol_ids = np.array([d.get('title', f"Mol_{i}") for i, d in enumerate(data)])
            self.smiles = np.array(['unknown'] * len(data))  # smiles not in msmollama2arr

            # Convert mol_meta into metadata fields if available
            self.metadata = {}
            if 'mol_meta' in data[0]:
                mol_meta = np.stack([d['mol_meta'] for d in data])
                self.metadata['mol_meta_mass'] = mol_meta[:, 0]
                self.metadata['mol_meta_atomicnum'] = mol_meta[:, 1]

        else:
            raise ValueError("Unsupported data format: must be DataFrame or list-of-dicts")

        # Data augmentation
        if data_augmentation:
            noise_factor = 0.01
            augmented_embeddings = (
                self.embeddings
                + np.random.normal(0, noise_factor, self.embeddings.shape).astype(np.float32)
            )
            self.embeddings = np.vstack([self.embeddings, augmented_embeddings])
            self.mol_ids = np.concatenate([self.mol_ids, [f"{mid}_aug" for mid in self.mol_ids]])
            self.smiles = np.concatenate([self.smiles, self.smiles])
            for col, values in self.metadata.items():
                self.metadata[col] = np.concatenate([values, values])
            print(f"Loaded {len(self.embeddings)} data points (with augmentation)")
        else:
            print(f"Loaded {len(self.embeddings)} data points")

        self.masks = np.ones((len(self.embeddings), self.embeddings.shape[1]), dtype=bool)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        metadata_dict = {col: values[idx] for col, values in self.metadata.items()}

        # x is returned as [2048, 1] to match your permute(0, 2, 1)
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32).unsqueeze(1)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)

        # y: label if present in metadata, otherwise dummy tensor
        if "label" in metadata_dict:
            y = torch.tensor(metadata_dict["label"], dtype=torch.float32)
        else:
            y = torch.tensor(0.0, dtype=torch.float32)

        return (
            self.mol_ids[idx],  # id
            x,                  # x
            mask,               # mask
            y,                  # y
            metadata_dict       # env
        )

    def get_embedding_dim(self):
        return self.embeddings.shape[1]

    def get_available_metadata(self):
        return list(self.metadata.keys())

	

class Mol_Dataset(Dataset):
	def __init__(self, data, precursor_type=False): 
		if precursor_type:
			data = self.filter_precursor_type(data, precursor_type)

		# generate mask
		for idx in range(len(data)): 
			mask = ~np.all(data[idx]['mol'] == 0, axis=1)
			data[idx]['mask'] = mask.astype(bool)

		self.data = data

	def __len__(self): 
		return len(self.data)

	def __getitem__(self, idx): 
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['mask'], self.data[idx]['env']

	def filter_precursor_type(self, data, precursor_type): 
		filtered_data = []
		for d in data:
			d_precursor_type = ','.join([str(int(i)) for i in d['env'][1:]])
			if d_precursor_type == precursor_type:
				filtered_data.append(d)
		return filtered_data



class MolRT_Dataset(Dataset): 
	def __init__(self, path): 
		with open(path, 'rb') as file: 
			self.data = pickle.load(file)
		print('Load {} data from {}'.format(len(self.data), path))

		# generate mask
		for idx in range(len(self.data)): 
			mask = ~np.all(self.data[idx]['mol'] == 0, axis=1)
			self.data[idx]['mask'] = mask.astype(bool)

	def __len__(self): 
		return len(self.data)

	def __getitem__(self, idx): 
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['mask'], self.data[idx]['rt']


class MolTox_LLMDataset(Dataset):
	def __init__(self, path):
		with open(path, 'rb') as file:
			self.data = pickle.load(file)
		print('Load {} data from {}'.format(len(self.data), path))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['label']

class MolTox_Dataset(Dataset):
	def __init__(self, path):
		with open(path, 'rb') as file:
			self.data = pickle.load(file)
		print('Load {} data from {}'.format(len(self.data), path))

	def __len__(self):
		return len(self.data)

	# def __getitem__(self, idx):
	# 	rec = self.data[idx]
	# 	return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['features'], self.data[idx]['label']

	def __getitem__(self, idx):
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['label'], self.data[idx]['mol_meta']


class MolCCS_Dataset(Dataset):
	def __init__(self, path): 
		with open(path, 'rb') as file: 
			self.data = pickle.load(file)
		print('Load {} data from {}'.format(len(self.data), path))

		# generate mask
		for idx in range(len(self.data)): 
			mask = ~np.all(self.data[idx]['mol'] == 0, axis=1)
			self.data[idx]['mask'] = mask.astype(bool)

	def __len__(self): 
		return len(self.data)

	def __getitem__(self, idx): 
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['mask'], self.data[idx]['ccs'], self.data[idx]['env']



class MolPRE_Dataset(Dataset): 
	def __init__(self, path): 
		with open(path, 'rb') as file: 
			data = pickle.load(file)
		
		self.data = []
		for d in data: 
			if 'mol' in d.keys(): 
				self.data.append(d)
		print('Load {} data from {}'.format(len(self.data), path))

		# generate mask
		for idx in range(len(self.data)): 
			mask = ~np.all(self.data[idx]['mol'] == 0, axis=1)
			self.data[idx]['mask'] = mask.astype(bool)

	def __len__(self): 
		return len(self.data)

	def __getitem__(self, idx): 
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['mask'], self.data[idx]['y']



class MolCSV_Dataset(Dataset): 
	def __init__(self, x, mode='path'): 
		assert mode in ['path', 'data']
		if mode == 'path': 
			with open(x, 'rb') as file: 
				self.data = pickle.load(file)
			print('Load {} data from {}'.format(len(self.data), x))
		elif mode == 'data': 
			self.data = x

		# generate mask
		for idx in range(len(self.data)): 
			mask = ~np.all(self.data[idx]['mol'] == 0, axis=1)
			self.data[idx]['mask'] = mask.astype(bool)

	def __len__(self): 
		return len(self.data)

	def __getitem__(self, idx): 
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['mask'], self.data[idx]['prop']

class MolCSV_Test_Dataset(Dataset): 
	def __init__(self, x, mode='path'): 
		assert mode in ['path', 'data']
		if mode == 'path': 
			with open(x, 'rb') as file: 
				self.data = pickle.load(file)
			print('Load {} data from {}'.format(len(self.data), x))
		elif mode == 'data': 
			self.data = x

		# generate mask
		for idx in range(len(self.data)): 
			mask = ~np.all(self.data[idx]['mol'] == 0, axis=1)
			self.data[idx]['mask'] = mask.astype(bool)

	def __len__(self): 
		return len(self.data)

	def __getitem__(self, idx): 
		return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx]['mask']