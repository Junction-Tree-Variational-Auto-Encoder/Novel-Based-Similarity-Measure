import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 

data = pd.read_csv('./data/train.txt', names=['SMILES'])

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19').cuda()

logP_values = []
for i in range(size(data['SMILES'])):
    logP, kl_div, _, _ = model.logP_molecule(data['SMILES'][i:i+1], 1)
    logP_values.append(logP.cpu().data.numpy())
out_df = pd.DataFrame(logP_values)
out_df.to_csv('./latent_space/logPvals/p_vals.txt')

out_mean = model.encode_to_latent(data['SMILES'])

outm_numpy = out_mean.cpu().data.numpy()
outm_df = pd.DataFrame(outm_numpy)

outv_np = out_var.cpu().data.numpy()
outv_df = pd.DataFrame(outv_np)

outm_df.to_csv('./latent_space/encodedZINC_9900to10000_mean.txt')
outv_df.to_csv('./latent_space/Var/encodedZINC_9900to10000_var.txt')

smiles = []
latent_rep = []
for i in range(5):
    nois_vec = []
    noise = np.random.normal(1,0.1, 56)
    noise = np.expand_dims(noise,axis = 0)
    noise = torch.from_numpy(noise).float().cuda()
    out_mean_noise= out_mean*noise

    z_t = out_mean_noise[0:1,0:28].cuda()
    z_mol = out_mean_noise[0:1,28:56].cuda()
    latent_rep.append(out_mean_noise)
    smiles.append(model.decode(z_t, z_mol, False))