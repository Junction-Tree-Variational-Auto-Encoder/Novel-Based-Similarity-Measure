import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

data = pd.read_csv('./data/train.txt', names=['SMILES'])

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19').cuda()

logP_values = model.logP_molecule(data['SMILES'], 1)
logP_values_df = pd.DataFrame(logP_values.cpu().data.numpy())
logP_values_df.to_csv('./latent_space/logP_values.txt')
