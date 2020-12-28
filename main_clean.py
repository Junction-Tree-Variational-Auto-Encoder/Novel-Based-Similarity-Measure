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

out_tensor = model.encode_from_smiles(inp['SMILES'][1400:2300])
out_numpy = out_tensor.cpu().data.numpy()
out_df = pd.DataFrame(out_numpy)

out_df.to_csv('./latent_space/encoded_1400to2300.txt', index=False)