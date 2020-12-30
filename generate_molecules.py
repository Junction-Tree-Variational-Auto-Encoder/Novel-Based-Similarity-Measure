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

####----------------Chris' kode-------------- ####
"""
out_tensor = model.encode_from_smiles(inp['SMILES'][0:900])
out_numpy = out_tensor.cpu().data.numpy()
out_df = pd.DataFrame(out_numpy)

out_df.to_csv('./latent_space/encodedZINC_0to900.txt')

"""
###-------------------------------------------####

import matplotlib.pyplot as plt
lat = pd.read_csv('./latent_space/encoded_ZINC.txt')
plt.hist(lat)
plt.hist(lat[['3']],bins = 1000)
mean_n1 = lat[['3']].mean()
st_n1  = lat[['3']].std()
# We decide to create latent representations around one standard deviation from the mean.
norm = np.random.normal(loc = nd1, scale = 0.5, size = 28)
norm = np.expand_dims(norm,axis = 0)
norm = torch.from_numpy(norm).float().cuda()

# Create normal distribution around a certain number nd1 : 
nd1 = st_n1
z = torch.randn(1, 28).cuda() # a random tensor of size (1, latent_size / 2)
sf = model.decode(norm,norm,False)


all_mols = []

smiles = []


norm = np.random.normal(loc = nd1, scale = 0.5, size = 28)
norm = np.expand_dims(norm,axis = 0)
norm = torch.from_numpy(norm).float().cuda()

x = [i for _ in range(2) for i in range(-1, 1)]
y = [i for i in range(-1, 1) for _ in range(2)]
label_float = [(norm[0][0].item()*(1 + xs*2), norm[0][1].item()*(1 + ys*2)) for xs, ys in zip(reversed(x), reversed(y))]
# label = ['%.3f'%(xs)+','+ '%.3f'%(ys) for xs, ys in label_float]#zip(reversed(x), reversed(y))]
z_labels = [norm.detach().clone() for _ in range(len(x))]

# Minor changes in the Tensor z to sample new molecules
for i, (xs, ys) in enumerate(label_float):
    z_labels[i][0][0] = xs
    z_labels[i][0][1] = ys


for zs in z_labels:
    smiles.append(model.decode(zs, zs, False))


out_df = pd.DataFrame(smiles)
out_df.to_csv('./New_mols/generated_molecules.txt')
















import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
print("loaded all packages")

inp = pd.read_csv('./data/train.txt', names=['SMILES'])
inp['INCHI'] = inp['SMILES'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
inp = inp.drop_duplicates(subset=['INCHI'], keep='first')
print("loaded data set")



from fast_molvae.sample import main_sample
#main_sample('./data/vocab.txt', './fast_molvae/vae_model/sample.txt', 'fast_molvae/vae_model/model.epoch-19', 10)

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19')

print("generating new molecules")
z = torch.randn(1, 56//2).cuda() # a random tensor of size (1, latent_size / 2)
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

from rdkit.Chem import Descriptors
logP_values_smiles = []
for i in range(len(smiles)):
    logP_values_smiles.append(
        Descriptors.MolLogP(MolFromSmiles(smiles[i])))

z = torch.randn(1, 56//2).cuda()

def kl_loss(z_vecs):
    batch_size = z_vecs.size(0)
    z_mean =torch.mean(z_vecs)
    z_var = torch.var(z_vecs)
    z_log_var = -torch.abs(z_var) #Following Mueller et al.
    kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
    epsilon = create_var(torch.randn_like(z_mean))
    z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
    return kl_loss






import pdb
import numpy as np
import torch
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple


Q = torch.distributions.Normal(mu,sigma) 

px = gaussian1.log_prob(x).exp() + gaussian2.log_prob(x).exp()

qx = Q.log_prob(x)

F.kl_div(qx,px)