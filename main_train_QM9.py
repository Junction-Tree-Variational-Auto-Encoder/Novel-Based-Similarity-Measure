import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

QM9 = pd.read_csv('./data/train_QM9.txt', header=0)\
    .rename(columns={'smiles':'SMILES'})
df_smiles = QM9['SMILES']
c_smiles = []

for ds in df_smiles:
    try:
        cs = Chem.CanonSmiles(ds)
        c_smiles.append(cs)
    except:
        print('Invalid SMILES:', ds)
print()

data_QM9 = pd.DataFrame(c_smiles, columns=['SMILES'])
data_QM9['INCHI'] = data_QM9['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
data_QM9 = data_QM9.drop_duplicates(subset=['INCHI'], keep='first')

img_QM9 = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in list(data_QM9['SMILES'])[:20]], molsPerRow=5,subImgSize=(500,500), legends=list(data_QM9['SMILES'])[:20])
img_QM9
open('./sh_and_pics/display_data_QM9.png','wb').write(img_QM9.data)

data_QM9['SMILES'].to_csv('./data/train_QM9.txt', header=False, index=False)

# Get QM9 Vocab
from fast_jtnn.mol_tree import main_mol_tree
main_mol_tree('./data/train_QM9.txt','./data/vocab_QM9.txt')

# Preprocess QM9 Data
from fast_molvae.preprocess import main_preprocess
main_preprocess('./data/train_QM9.txt','./fast_molvae/processed_QM9/', num_splits=100)

# Train Model with QM9 Data
from fast_molvae.vae_train import main_vae_train
model = main_vae_train('./fast_molvae/processed/', './data/vocab.txt', './fast_molvae/vae_model/')

## Training of the model with the QM9 dataset was abandoned after this
