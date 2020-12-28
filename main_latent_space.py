import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

inp = pd.read_csv('./data/train.txt', names=['SMILES'])

inp['INCHI'] = inp['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
inp = inp.drop_duplicates(subset=['INCHI'], keep='first')
''
img = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in list(inp['SMILES'])[20:30]], molsPerRow=5,
            subImgSize=(500,500), 
            legends=list(inp['SMILES'])[20:30])
img

from fast_molvae.sample import main_sample
# main_sample('./data/vocab.txt', './fast_molvae/vae_model/sample.txt', 'fast_molvae/vae_model/model.epoch-19', 100)
out = pd.read_csv('./fast_molvae/vae_model/sample.txt', names=['SMILES'])
out['INCHI'] = out['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
out = out.drop_duplicates(subset=['INCHI'], keep='first')

img = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in list(out['SMILES'])[:10]], molsPerRow=5,subImgSize=(500,500), legends=list(out['SMILES'])[:10])
img

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19')

z = torch.randn(1, 56//2).cuda() # a random tensor of size (1, latent_size / 2)

print('Random Smile: ', model.decode(z, z, True))

import matplotlib.pyplot as plt


x = [i for _ in range(7) for i in range(-3, 4)]
y = [i for i in range(-3, 4) for _ in range(7)]
label_float = [(z[0][0].item()*(1 + xs*2), z[0][1].item()*(1 + ys*2)) for xs, ys in zip(reversed(x), reversed(y))]
# label = ['%.3f'%(xs)+','+ '%.3f'%(ys) for xs, ys in label_float]#zip(reversed(x), reversed(y))]
z_labels = [z.detach().clone() for _ in range(len(x))]

# Minor changes in the Tensor z to sample new molecules
for i, (xs, ys) in enumerate(label_float):
    z_labels[i][0][0] = xs
    z_labels[i][0][1] = ys

smiles = []
for zs in z_labels:
    smiles.append(model.decode(zs, zs, False))
img = Draw.MolsToGridImage([Chem.MolFromSmiles(mol) for mol in smiles], molsPerRow=7,subImgSize=(250,250))
img


from fast_bo.gen_latent import main_gen_latent
target_input = pd.read_csv('./fast_bo/descriptors/targets.txt', names=['Target'])


# Latent space distance
a = model.encode_from_smiles(inp['SMILES'].iloc[0:1000])

def article_sim(molecules_in_smiles_series)
    Latent_Rep = model.encode_from_smiles(molecules_in_smiles_series)
    logP_values = []
    for i in range(len(molecules_in_smiles_series)):
        logP_values.append(
            Descriptors.MolLogP(MolFromSmiles(molecules_in_smiles_series[i])))
    for i in range(len(molecules_in_smiles_series)):
        for j in range(len(molecules_in_smiles_series)):

    EUClid = 1/(torch.sqrt(torch.sum((test[0]-test[1])**2)))
    return EUClid

import numpy as np
import matplotlib.pyplot as plt

x = np.array(logP_values)
y = np.array(logP_QM9)
# the histogram of the data
plot = plt.hist(x, bins=50, label='ZINC')
plot = plt.hist(y, bins=50, label='QM9')
plt.legend()
plt.xlabel('logP(x)')
plt.ylabel('# Observations')
