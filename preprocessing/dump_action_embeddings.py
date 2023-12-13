from rdkit import Chem
import numpy as np
import pandas as pd
import tqdm
from ChemRL import ChemRlEnv
import pickle

# Load csv
csv_path = "datasets/my_uspto/action_dataset-filtered.csv"
dataset = pd.read_csv(csv_path, index_col=0)
print(dataset.shape)

# Load env
env = ChemRlEnv()
action_embeddings = []
failures = []

# Get all embeddings
for i in tqdm.tqdm(range(dataset.shape[0])):
    row = dataset.iloc[i][["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]]
    try:
        action_embeddings.append(env._action_embedding(row))
    except Exception as e:
        for sub in ["rsub", "psub"]:
            if Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(row[sub]))) == row[sub]:
                row[sub] = row[sub] + "[H]"
        action_embeddings.append(env._action_embedding(row))
        failures.append(i)


# Create hash for embeddings for fast search
action_embedding_hash = list(map(lambda x: hash(" ".join(map(str, x))), action_embeddings))
action_embeddings = np.array(action_embeddings)
action_embedding_hash = np.array(action_embedding_hash)

# Remove hash collisions (some actions that weren't detected as different before)
x, y = np.unique(action_embedding_hash, return_counts=True)
collision_hashes = x[y!=1]
idx_arr = np.isin(action_embedding_hash, collision_hashes, invert=True)

for coll in collision_hashes:
    idx_arr[np.where(action_embedding_hash==coll)[0][0]] = True

assert idx_arr.sum() == action_embedding_hash.shape[0] - y[y!=1].sum() + y[y!=1].shape[0]


# Set embedding hash as index
dataset.index = action_embedding_hash

dataset = dataset.loc[idx_arr]

dataset.to_csv(csv_path)

# Dump the hash->embedding dict
hash_to_embedding_map = {action_embedding_hash[i]: action_embeddings[i] for i in range(len(action_embeddings))}
pickle.dump(hash_to_embedding_map, open("datasets/my_uspto/action_embeddings.pickle", 'wb'))