import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

inp = pd.read_csv('./data/train.txt', names=['SMILES'])
inp['INCHI'] = inp['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
inp = inp.drop_duplicates(subset=['INCHI'], keep='first')

img = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in list(inp['SMILES'])[:20]], molsPerRow=5,subImgSize=(500,500), legends=list(inp['SMILES'])[:20])
img

from fast_molvae.sample import main_sample
# main_sample('./data/vocab.txt', './fast_molvae/vae_model/sample.txt', 'fast_molvae/vae_model/model.epoch-19', 100)
out = pd.read_csv('./fast_molvae/vae_model/sample.txt', names=['SMILES'])
out['INCHI'] = out['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
out = out.drop_duplicates(subset=['INCHI'], keep='first')

img = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in list(out['SMILES'])[:20]], molsPerRow=5,subImgSize=(500,500), legends=list(out['SMILES'])[:20])
img

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19')

z = torch.randn(1, 56//2).cuda()
print('Random Smile: ', model.decode(z,z,False))

#Latent space distance
a = model.encode_from_smiles(inp['SMILES'].iloc[0:10])