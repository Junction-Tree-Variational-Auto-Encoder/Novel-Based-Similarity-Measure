import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 



data = pd.read_csv('./data/train.txt', names=['SMILES'])

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19').cuda()

out_latent = model.encode_to_latent(data['SMILES'][3:4])

smiles = []
latent_space = []


for i in range(1):
    noise = np.random.normal(1,0.1, 56)
    noise = np.expand_dims(noise,axis = 0)
    noise = torch.from_numpy(noise).float().cuda()
    out_latent_noise=out_latent*noise

    z_t = out_latent_noise[0:1,0:28].cuda()
    z_mol = out_latent_noise[0:1,28:56].cuda()
    latent_space.append(out_latent_noise)
    smiles.append(model.decode(z_t, z_mol, False))


# exporting the molecules

out_df = pd.DataFrame(smiles, columns = ['SMILES'])
out_df.to_csv('./generated_moleculesX.txt')


