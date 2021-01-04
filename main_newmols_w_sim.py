import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 

# New Molecule

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
out_df.to_csv('./New_mols/generated_moleculesX.txt')



## Similarity tanimoto
new_mol = out_df['SMILES'][0]
data_smiles = pd.read_csv('./data/train.txt', names=['SMILES'])

orig_mol = data_smiles['SMILES'][3]
from rdkit import DataStructs, Chem
new_mol_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(new_mol))

orig_mol_fp= Chem.RDKFingerprint(Chem.MolFromSmiles(data_smiles['SMILES'][3]))
tan_sim = DataStructs.FingerprintSimilarity(orig_mol_fp, new_mol_fp)


# Similarity euclidean

data_latent = pd.read_csv('./latent_space/encoded_ZINC_mean.txt').drop(columns={'Unnamed: 0'})


def euclid(x,y):
    sqrtsum= 0
    xx = x.cpu().data.numpy()
    yy = y.cpu().data.numpy()
    for i in range(len(xx)):
        sqrtsum += (xx[0,i]-yy[0,i])**2
    EUClid = 1 / ( 1 + np.sqrt(sqrtsum))
    return EUClid


chosen_mol = out_latent_noise
euc_dist = euclid( torch.tensor(data_latent.iloc[3,:]).cuda().unsqueeze_(0) ,  chosen_mol  )

data_smiles['Tanimoto_Similarity'] = tan_sim
data_smiles['Euclidian_distance'] = euc_list

data_smiles.to_csv('./latent_space/data_smiles_with_tanimoto_and_euclidian_to_new_mol.txt')