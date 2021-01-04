import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 

new_mol = '[NH2+]=C1CC2(CN1)C1=[NH+]CCCCC1C21CS2(CCS1)CCS1(CCCN1)CC2'





data_smiles = pd.read_csv('./data/train.txt', names=['SMILES'])


from rdkit import DataStructs, Chem
original_mol = Chem.RDKFingerprint(Chem.MolFromSmiles(new_mol))
#original_mol = Chem.RDKFingerprint(Chem.MolFromSmiles(data_smiles['SMILES'][3]))
ms = []
for i in range(len(data_smiles)):
    ms.append(Chem.MolFromSmiles(data_smiles['SMILES'][i]))

#ms = [Chem.MolFromSmiles(smiles[0]), Chem.MolFromSmiles(smiles[1]), Chem.MolFromSmiles(smiles[2]), Chem.MolFromSmiles(smiles[3]), Chem.MolFromSmiles(smiles[4])]
fps = [Chem.RDKFingerprint(x) for x in ms]
tan_sim = []
for i in range(len(fps)): 
    tan_sim.append(DataStructs.FingerprintSimilarity(original_mol,fps[i]))



data = pd.read_csv('./latent_space/encoded_ZINC_mean.txt').drop(columns={'Unnamed: 0'})

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

data_smiles.to_csv('./latent_space/shaakabraah/data_smiles_with_tanimoto_and_euclidian.txt')


data_smiles= data_smiles.sort_values(by =  'Tanimoto_Similarity', ascending = False)
data_smiles_copy = data_smiles
data_smiles_copy = data_smiles_copy.sort_values(by = 'Euclidian_distance', ascending = False)
import matplotlib.pyplot as plt

ax = plt.figure()
plt.scatter(np.arange(0,len(data_smiles)), data_smiles_copy['Tanimoto_Similarity'], s = 1, label = 'Tanimoto similarity')
plt.scatter(np.arange(0,len(data_smiles)),data_smiles_copy['Euclidian_distance'], s = 4, label = 'Euclidian distance')
plt.title('Similarities')
plt.ylabel('Tanimoto Coefficient/ Euclidian distance')
plt.xlabel('Molecules')
plt.legend()

ax.savefig('tarx.png')


