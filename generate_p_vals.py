import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 


data_tan_eu = pd.read_csv('./latent_space/shaakabraah/data_smiles_with_tanimoto_and_euclidian.txt').drop(columns={'Unnamed: 0'})
data_p_vals = pd.read_csv('./latent_space/P_values.txt').drop(columns={'Unnamed: 0'}).rename(columns= {'0':'P_values'})


data_p_vals['logp'] = np.log(data_p_vals['P_values'])

data_p_vals['Tanimoto'] = data_tan_eu['Tanimoto_Similarity']
data_p_vals['Euclidean_distance'] = data_tan_eu['Euclidian_distance']


#(within 5% of value)
cut_off_value = 0.05
orig_mol =  data_p_vals.iloc[3,1]

#Difference in %. where 0% is close and 100% is completely different
data_p_vals['Cut off'] =np.abs(data_p_vals['logp'] -  orig_mol) /  orig_mol

data_p_cut = data_p_vals[(data_p_vals['Cut off']<=  cut_off_value)]


import matplotlib.pyplot as plt

##plt.scatter(np.arange(0,len(data_p_vals)), data_p_vals['logp'], s = 1, label = 'values within %')
#plt.plot(cut_off_value)
plt.hist(data_p_cut['logp'], bins= 100)
plt.title('Cut off')
plt.ylabel('e')
plt.xlabel('Molecules')
plt.legend()
