import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 


data_tan_eu = pd.read_csv('./latent_space/data_smiles_with_tanimoto_and_euclidian.txt').drop(columns={'Unnamed: 0'})



data_p_vals = pd.read_csv('./latent_space/logP_values.txt').drop(columns={'Unnamed: 0'}).rename(columns= {'0':'logp'})

import matplotlib.pyplot as plt
fig = plt.figure()
plt.hist(data_p_vals['logp'], bins= 100)
plt.ylabel('Molecules')
plt.xlabel('log p(x)')
plt.xlim([3,8])
plt.legend()
fig.savefig('./sh_and_pics/log_p_his.png')

data_p_vals['Tanimoto'] = data_tan_eu['Tanimoto_Similarity']
data_p_vals['Euclidean_distance'] = data_tan_eu['Euclidian_distance']


#
cut_off_value = 0.50
orig_mol_logP =  data_p_vals.iloc[3,1]

#
data_p_vals['cut_off'] =np.abs(data_p_vals['logp'] -  orig_mol_logP)
data_p_vals['novel_sim'] = data_p_vals['Euclidean_distance']
data_p_vals['novel_sim'].loc[data_p_vals['cut_off'] > cut_off_value] = 0
