import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

data = pd.read_csv('./data/train.txt', names=['SMILES'])

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19')

#out_tensor = model.encode_from_smiles(data['SMILES'][:10])

out_vecs, out_mean, out_var = model.encode_test(data['SMILES'][:300])


out_numpy = out_tensor.cpu().data.numpy()
out_df = pd.DataFrame(out_numpy)

out_df.to_csv('./latent_space/encoded_ZINC.txt')
