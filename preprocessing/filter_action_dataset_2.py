from rdkit import Chem
import numpy as np
import pandas as pd
from IPython.display import display
import tqdm

dataset = pd.read_csv("datasets/my_uspto/action_dataset-filtered.csv", index_col=0)

mts = Chem.MolToSmiles
mfs = Chem.MolFromSmiles

def molecule_equality(m1, m2):
    if isinstance(m1, str):
        m1 = mfs(m1)
    if isinstance(m2, str):
        m2 = mfs(m2)
    m1 = mfs(mts(m1))
    m2 = mfs(mts(m2))
    
    if m1 is None or m2 is None:
        return False
    return (mts(m1, 1) == mts(m2, 1)) or (m1.HasSubstructMatch(m2) and m2.HasSubstructMatch(m1))

import os
os.environ["MAIN_DIR"] = ""
from action_utils import apply_action

l = np.array([True] * dataset.shape[0])

for i in tqdm.tqdm(range(dataset.shape[0])):
    row = dataset.iloc[i]
    reactant = mfs(row["reactants"])
    action = row[["rsub", "rcen", "rsig", "rbond", "rsig_cs_indices", "psub", "pcen", "psig", "pbond", "psig_cs_indices"]]
    product = mfs(row["products"])
    
    try:
        p2 = apply_action(reactant, *action, use_advanced_algo=False)
        p3 = apply_action(reactant, *action, use_advanced_algo=True)
    except:
        continue
    
    if not molecule_equality(product, p2) or not molecule_equality(product, p3):
        l[i] = False

dataset["action_works"] = (dataset["action_works"].values & l)
dataset.to_csv("datasets/my_uspto/action_dataset-filtered.csv")