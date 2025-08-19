import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# >>>           encoder part           <<<
# ----------------------------------------
class Encoder(nn.Module):
	def __init__(self, in_dim, layers, emb_dim, point_num, k):
		super(Encoder, self).__init__()
		self.emb_dim = emb_dim
		self.hidden_layers = nn.ModuleList([MolConv3(in_dim=in_dim, out_dim=layers[0], point_num=point_num, k=k, remove_xyz=True)])
		for i in range(1, len(layers)):
			if i == 1:
				self.hidden_layers.append(MolConv3(in_dim=layers[i-1], out_dim=layers[i], point_num=point_num, k=k, remove_xyz=False))
			else:
				self.hidden_layers.append(MolConv3(in_dim=layers[i-1], out_dim=layers[i], point_num=point_num, k=k, remove_xyz=False))

		self.conv = nn.Sequential(nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=False),
								nn.LayerNorm((emb_dim, point_num)),
								nn.LeakyReLU(negative_slope=0.2))

	def forward(self, x: torch.Tensor,
						idx_base: torch.Tensor,
						mask: torch.Tensor) -> torch.Tensor:
		xs = []
		for i, hidden_layer in enumerate(self.hidden_layers):
			if i == 0:
				tmp_x = hidden_layer(x, idx_base, mask)
			else:
				tmp_x = hidden_layer(xs[-1], idx_base, mask)
			xs.append(tmp_x)

		x = torch.cat(xs, dim=1) # torch.Size([batch_size, emb_dim, point_num])
		x = self.conv(x)

		# Apply the mask: Set padding points to a very low value for max pooling and zero for average pooling
		mask_expanded = mask.unsqueeze(1).expand_as(x) # [batch_size, emb_dim, point_num]
		x_masked_max = x.masked_fill(~mask_expanded, float('-inf')) # Replace padding with -inf for max pooling
		x_masked_avg = x.masked_fill(~mask_expanded, 0.0) # Replace padding with 0 for average pooling

		# Max pooling along the third dimension
		max_pooled = torch.max(x_masked_max, dim=2)[0] # [batch_size, emb_dim]

		# Average pooling along the third dimension
		# Count the valid (non-padding) points for each position
		valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=0.1) # Avoid division by zero
		avg_pooled = x_masked_avg.sum(dim=2) / valid_counts # [batch_size, emb_dim]

		x = max_pooled + avg_pooled
		return x


# -------------------------------------------------------------------------
# >>>                             3DMol_tox                             <<<
# -------------------------------------------------------------------------
class MolnetTox_bin(nn.Module):
	def __init__(self, config):
		super(MolnetTox_bin, self).__init__()
		self.add_num = config['add_num']
		self.encoder = Encoder(in_dim=int(config['in_dim']),
							   layers=config['encode_layers'],
							   emb_dim=int(config['emb_dim']),
							   k=int(config['k']))
		self.decoder = MSDecoder(in_dim=int(config['emb_dim'] + config['add_num']),
								 layers=config['decode_layers'],
								 out_dim=3,
								 dropout=config['dropout'])

		self.classifier = nn.Linear(3, 1)

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
			elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(mean=0.0, std=1.0)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x: torch.Tensor,
				env: torch.Tensor,
				idx_base: torch.Tensor) -> torch.Tensor:
		'''
        Input:
            x:      point set, torch.Size([batch_size, 14, atom_num])
            env:    experimental condition
            idx_base:   idx for local knn
        '''
		x = self.encoder(x, idx_base)

		if self.add_num == 1:
			x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
		elif self.add_num > 1:
			x = torch.cat((x, env), 1)

		x = self.decoder(x)

		logits = self.classifier(x)

		return logits

class MolNet_LLM(nn.Module):
	def __init__(self, config):
		super(MolNet_LLM, self).__init__()
		# Configuration
		self.input_dim = config['input_dim']  # e.g. 2048
		self.hidden_dims = config['hidden_dims']  # e.g. [64, 64, 128, ...]
		self.dropout = config.get('dropout', 0.2)

		# Input layer
		self.layers = nn.ModuleList([
			nn.Sequential(
				nn.Linear(self.input_dim, self.hidden_dims[0]),
				nn.LayerNorm(self.hidden_dims[0]),
				nn.LeakyReLU(0.2),
				nn.Dropout(self.dropout)
			)
		])

		# Hidden layers
		for i in range(len(self.hidden_dims) - 1):
			self.layers.append(
				nn.Sequential(
					nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
					nn.LayerNorm(self.hidden_dims[i + 1]),
					nn.LeakyReLU(0.2),
					nn.Dropout(self.dropout)
				)
			)

		# Output layer (no sigmoid — raw logit)
		self.output = nn.Linear(self.hidden_dims[-1], 1)

		# Weight initialization
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.LayerNorm):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		out = self.output(x)  # Shape: (B, 1), raw logit
		return out


