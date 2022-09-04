'''
This code reads simulator_dataset.csv and extracts all unique mols.
'''

import pandas as pd
from rdkit import Chem
from utils import get_mol_certificate
import tqdm

df = pd.read_csv("datasets/my_uspto/simulator_dataset.csv")
mols = df["reactants"]

# certi to mol(df index) dict
d = {}

for i, mol in tqdm.tqdm(enumerate(mols)):
    certi = get_mol_certificate(Chem.MolFromSmiles(mol))
    if certi not in d:
        d[certi] = [i]
    else:
        d[certi].append(i)

# get the unique mol indices
unique_idx_list = [d[key][0] for key in d]
print("Unique mols:", len(unique_idx_list))

# dump mols
mols[unique_idx_list].to_pickle("datasets/my_uspto/unique_start_mols.pickle")