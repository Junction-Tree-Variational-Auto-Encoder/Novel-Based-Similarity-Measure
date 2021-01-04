import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

data = pd.read_csv('./data/train.txt', names=['SMILES'])

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19').cuda()

out_latent = model.encode_to_latent(data['SMILES'])
out_df = pd.DataFrame(out_latent.cpu().data.numpy())
out_out.to_csv('./latent_space/encoded_ZINC_data.txt')