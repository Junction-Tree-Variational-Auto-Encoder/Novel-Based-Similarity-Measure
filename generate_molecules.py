import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

inp = pd.read_csv('./data/train.txt', names=['SMILES'])

#inp['INCHI'] = inp['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
#inp = inp.drop_duplicates(subset=['INCHI'], keep='first')

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19')

####----------------Chris' kode-------------- ####
"""
out_tensor = model.encode_from_smiles(inp['SMILES'][0:900])
out_numpy = out_tensor.cpu().data.numpy()
out_df = pd.DataFrame(out_numpy)

out_df.to_csv('./latent_space/encodedZINC_0to900.txt')

"""
###-------------------------------------------####
# Create normal distribution around a certain number nd1 : 
import numpy as np
nd1 = 1.2

norm_dis = np.random.normal(loc = nd1, scale = 0.1, size = 10)
#z = torch.randn(1, 56//2).cuda() # a random tensor of size (1, latent_size / 2)
out_df = model.decode(norm_dis, norm_dis, False)
out_df = pd.DataFrame(out_df)
out_df.to_csv('./new_mols/generated_molecules.txt')

