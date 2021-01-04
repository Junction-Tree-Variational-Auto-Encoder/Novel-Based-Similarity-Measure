import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np

data = pd.read_csv('./data/train.txt', names=['SMILES'])

zinc_img = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in list(data['SMILES'])[18:24]], molsPerRow=2,subImgSize=(200,200), legends=list(data['SMILES'])[18:24])
zinc_img

from fast_jtnn.mol_tree import main_mol_tree
main_mol_tree('./data/train.txt', './data/vocab.txt')

from fast_molvae.preprocess import main_preprocess
main_preprocess('./data/train.txt', './fast_molvae/processed/', num_splits=100)

from fast_molvae.vae_train import main_vae_train
model = main_vae_train('./fast_molvae/processed/', './data/vocab.txt', './fast_molvae/vae_model/')