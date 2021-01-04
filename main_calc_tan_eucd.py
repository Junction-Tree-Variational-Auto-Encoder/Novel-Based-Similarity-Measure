import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 




data_smiles = pd.read_csv('./data/train.txt', names=['SMILES'])

new_mol = data_smiles['SMILES'][4]


#### Create Tanimoto sim for chosen molecule

from rdkit import DataStructs, Chem
original_mol = Chem.RDKFingerprint(Chem.MolFromSmiles(new_mol))
ms = []
for i in range(len(data_smiles)):
    ms.append(Chem.MolFromSmiles(data_smiles['SMILES'][i]))

fps = [Chem.RDKFingerprint(x) for x in ms]
tan_sim = []
for i in range(len(fps)): 
    tan_sim.append(DataStructs.FingerprintSimilarity(original_mol,fps[i]))


####### Create Euclidean distance for for ZINC to chosen molecule


data = pd.read_csv('./latent_space/encoded_ZINC_data.txt').drop(columns={'Unnamed: 0'})

def euclid(x,y):
    sqrtsum= 0
    xx = x.cpu().data.numpy()
    yy = y.cpu().data.numpy()
    for i in range(len(xx)):
        sqrtsum += (xx[0,i]-yy[0,i])**2
    EUClid = 1 / ( 1 + np.sqrt(sqrtsum))
    return EUClid


euc_list = []
chosen_mol = torch.tensor(data.iloc[3,:])
chosen_mol = chosen_mol.cuda().unsqueeze_(0)
r, j = data.shape
for i in range(len(data)):
    euc_list.append(  euclid( torch.tensor(data.iloc[i,:]).cuda().unsqueeze_(0) ,  chosen_mol  ))



data_smiles['Tanimoto_Similarity'] = tan_sim
data_smiles['Euclidian_distance'] = euc_list

data_smiles.to_csv('./latent_space/data_smiles_with_tanimoto_and_euclidian.txt')
