import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 
inp = pd.read_csv('./data/train.txt', names=['SMILES'])


from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19')


import matplotlib.pyplot as plt
lat = pd.read_csv('./latent_space/encoded_ZINC.txt').drop(columns = {'Unnamed: 0'})


mean_n1 = lat[['3']].mean()
st_n1  = lat[['3']].std()
nd1 = st_n1
# We decide to create latent representations around one standard deviation from the mean.
norm = np.random.normal(loc = nd1, scale = 0.5, size = 28)
norm = np.expand_dims(norm,axis = 0)
norm = torch.from_numpy(norm).float().cuda()

# Create normal distribution around a certain number nd1 : 
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




inp = pd.read_csv('./New_mols/generated_molecules.txt', names=['SMILES'])


out_tensor = model.encode_from_smiles(inp)
out_numpy = out_tensor.cpu().data.numpy()
out_df = pd.DataFrame(out_numpy)

out_df.to_csv('./New_mols/generated_latent_for_new_mols.txt')








import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
import numpy as np 



data = pd.read_csv('./data/train.txt', names=['SMILES'])

from fast_molvae.sample import load_model
model = load_model('./data/vocab.txt', './fast_molvae/vae_model/model.epoch-19').cuda()

out_vecs, out_mean, out_var = model.encode_test(data['SMILES'][:5])

smiles = []
latent_space = []


for i in range(1):
    nois_vec = []
    noise = np.random.normal(1,0.1, 56)
    noise = np.expand_dims(noise,axis = 0)
    noise = torch.from_numpy(noise).float().cuda()
    out_mean_noise= out_mean*noise

    z_t = out_mean_noise[0:1,0:28].cuda()
    z_mol = out_mean_noise[0:1,28:56].cuda()
    latent_space.append(out_mean_noise).cpu().data.numpy()
    smiles.append(model.decode(z_t, z_mol, False))




out_df = pd.DataFrame(smiles, columns = ['SMILES'])
out_latent = pd.DataFrame(latent_space, columns = ['Latent space'])
out_df['latent_space'] = out_latent
out_latent.to_csv('./New_mols/generated_molecules3.txt')


plt.hist(lat[['3']],bins = 1000)
plt.scatter(lat[['3']],lat[['5']], s= 4)





from rdkit import DataStructs, Chem

original_mol = Chem.RDKFingerprint(Chem.MolFromSmiles(data['SMILES'][0]))

ms = [Chem.MolFromSmiles(smiles[0]), Chem.MolFromSmiles(smiles[1]), Chem.MolFromSmiles(smiles[2]), Chem.MolFromSmiles(smiles[3]), Chem.MolFromSmiles(smiles[4])]
fps = [Chem.RDKFingerprint(x) for x in ms]
tan_sim = []
for i in range(len(fps)): 
    tan_sim.append(DataStructs.FingerprintSimilarity(original_mol,fps[i]))



def euclid(x,y):
    sqrtsum= 0
    xx = x.cpu().data.numpy()
    yy = y.cpu().data.numpy()
    for i in range(len(xx)):
        sqrtsum += (xx[0,i]-yy[0,i])**2
    EUClid = 1 / ( 1 + np.sqrt(sqrtsum))
    return EUClid

print(euclid(z_t,z_mol))
