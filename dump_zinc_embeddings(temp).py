import glob
from torchdrug import data
import numpy as np
import pandas as pd
import tqdm
import torch

model_list = glob.glob("models/zinc*")

print(model_list)

def mol_embedding(emb_model, smiles):
    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        emb = emb_model(mol, mol.node_feature.float())["graph_feature"]
    except Exception as e:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        emb = emb_model(mol, mol.node_feature.float())["graph_feature"]
    return emb.detach().cpu()[0]

def atom_embedding(emb_model, smiles, idx):
    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        emb = emb_model(mol, mol.node_feature.float())["node_feature"][idx]
    except Exception as e:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        emb = emb_model(mol, mol.node_feature.float())["node_feature"][idx]
    return emb.detach().cpu()

def action_embedding(emb_model, action):
    rsub, rcen, rsig, _, psub, pcen, psig, __ = action
    embedding = np.concatenate([
#                         mol_embedding(rsub), 
                        atom_embedding(emb_model, rsig, rcen) / 5, 
                        mol_embedding(emb_model, rsig), 
#                         mol_embedding(psub), 
                        atom_embedding(emb_model, psig, pcen) / 5, 
                        mol_embedding(emb_model, psig)
                    ])
    return embedding



action_dataset = pd.read_csv("datasets/my_uspto/action_dataset-filtered.csv", index_col=0)

action_dataset = action_dataset.loc[action_dataset["reactant_works"] & action_dataset["reactant_tested"] & action_dataset["action_tested"] & action_dataset["action_works"]]
action_dataset.shape

action_dataset = action_dataset[["rsub", "rcen", "rsig", "rbond", "psub", "pcen", "psig", "pbond"]]
print(action_dataset.shape)

for model_path in model_list:
    print("Using model", model_path)

    emb_model = torch.load(model_path)
    
    action_embeddings = []
    for i in tqdm.tqdm(range(action_dataset.shape[0])):
        action_embeddings.append(action_embedding(emb_model, action_dataset.iloc[i]))
    action_embeddings = np.stack(action_embeddings)
    print(action_embeddings.shape)

    np.save(model_path.replace("models", "datasets/my_uspto/supervised_zinc_gin").replace("pth", "npy"), action_embeddings)