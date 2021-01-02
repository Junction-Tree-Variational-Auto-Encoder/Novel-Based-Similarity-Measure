import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch

data = pd.read_csv('./data/train.txt', names=['SMILES'])

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19')

out_latent = model.encode_to_latent(data['SMILES'][9900:10000])
out_out = pd.DataFrame(out_latent.cpu().data.numpy())
out_out.to_csv('./latent_space/encodedZINC_9900to10000_mean.txt')


test = pd.read_csv('./latent_space/encoded_ZINC.txt').drop(columns={'Unnamed: 0'})

z_t = out_mean[:1,0:28].cuda()
z_mol = out_mean[:1,28:56].cuda()

testing = model.decode(z_t, z_mol, True)

model_out = []
for i in range(1, 20+1):
    z_t = out_mean[i-1:i,0:28].cuda()
    z_mol = out_mean[i-1:i,28:56].cuda()
    out = model.decode(z_t, z_mol, False)
    model_out.append(out)

out_numpy = out_tensor.cpu().data.numpy()
out_df = pd.DataFrame(out_numpy)

out_df.to_csv('./latent_space/encoded_ZINC.txt')
