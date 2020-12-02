import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

#inp_QM9 = pd.read_csv('./data/train_QM9.txt', names=['SMILES'])
#inp_QM9['INCHI'] = inp_QM9['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
#inp_QM9 = inp_QM9.drop_duplicates(subset=['INCHI'], keep='first')

#img_QM9 = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in list(inp_QM9['SMILES'])[:20]], molsPerRow=5,subImgSize=(500,500), legends=list(inp_QM9['SMILES'])[:20])
#img_QM9

# Get QM9 Vocab
from fast_jtnn.mol_tree import main_mol_tree
main_mol_tree('./data/train_QM9.txt','./data/vocab_QM9.txt')

# Preprocess QM9 Data
from fast_molvae.preprocess import main_preprocess
main_preprocess('./data/train_QM9.txt','./fast_molvae/processed_QM9/', num_splits=100)

# Train model
from fast_molvae.vae_train import main_vae_train
model = main_vae_train('./fast_molvae/processed_QM9/', './data/vocab_QM9.txt', './fast_molvae/vae_model_QM9/')
