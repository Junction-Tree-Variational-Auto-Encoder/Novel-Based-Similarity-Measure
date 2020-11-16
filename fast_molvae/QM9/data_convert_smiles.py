import numpy as np
import pandas as pd
import os

dir = 'C:\\Users\\Chris\\Documents\\GitHub\\JTVAE-on-Molecular-Structures\\data\\QM9'
data = pd.read_csv(dir + '\\gdb9_prop_smiles.csv')
output = data['smiles']
output.to_csv('train.txt', sep =' ', index=False)
