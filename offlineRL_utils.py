from rdkit import Chem
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import tqdm
import json
import numpy as np
import itertools, functools
from tabulate import tabulate
import torch.nn as nn
import torch
from torchdrug import data
from action_utils import *
from multiprocessing import Pool
import time

from action_utils import dataset as action_dataset
action_dataset = action_dataset[["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]]

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, num_hidden=1, hidden_size=50):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            self.hidden_layers.append(nn.ReLU())
            
        self.last_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.last_layer(out)
        return out

def molecule_from_smile(smile):
    try:
        mol = data.Molecule.from_smiles(smile, atom_feature="pretrain", bond_feature="pretrain")
    except Exception as e:
        mol = data.Molecule.from_smiles(smile, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
    return mol

def get_mol_embedding(model, smiles):
    # deepchem - attribute masking
    if isinstance(smiles, str):
        mol = molecule_from_smile(smiles)
    elif isinstance(smiles, list) or isinstance(smiles, pd.Series):
        mol = list(map(molecule_from_smile, smiles))
        mol = data.Molecule.pack(mol)
    else:
        mol = smiles
    mol = mol.to(model.device)
    emb = model(mol, mol.node_feature.float())["graph_feature"]
    return emb.detach()

def get_atom_embedding(model, smiles, idx):
    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        emb = model(mol, mol.node_feature.float())["node_feature"][idx]
    except Exception as e:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        emb = model(mol, mol.node_feature.float())["node_feature"][idx]
    return emb.detach()

def get_action_embedding(model, action_df):
    rsub, rcen, rsig, _, psub, pcen, psig, __ = [action_df[c] for c in action_df.columns]
    embedding = torch.concatenate([get_mol_embedding(model, rsig), get_mol_embedding(model, psig)], axis=1)
    return embedding

def get_action_embedding_from_packed_molecule(model, rsig, psig):
    embedding = torch.concatenate([get_mol_embedding(model, rsig), get_mol_embedding(model, psig)], axis=1)
    return embedding

def get_action_dataset_embeddings(model, action_rsigs, action_psigs, batch_size=2048):
    # batch_size = 2048
    action_embeddings = []
    for i in tqdm.tqdm(range(0, action_rsigs.batch_size, batch_size)):
        batch_rsig = action_rsigs[i:min(i+batch_size, action_rsigs.batch_size)]
        batch_psig = action_psigs[i:min(i+batch_size, action_psigs.batch_size)]
        action_embeddings.append(get_action_embedding_from_packed_molecule(model, batch_rsig, batch_psig))
    action_embeddings = torch.concatenate(action_embeddings)
    return action_embeddings

def get_emb_indices_and_correct_idx(row, no_correct_idx=False):
    if isinstance(row, tuple): # For pandas iterrows
        row = row[1]
    
    # Applicable indices
    applicable_actions_df = get_applicable_actions(Chem.MolFromSmiles(row["reactant"]))
    if applicable_actions_df.shape[0] == 0:
        # If there are no applicable actions detected (rdkit problems)
        if no_correct_idx is False:
            indices_used_for_data = np.where((action_dataset.index == row.name))[0]
            correct_applicable_idx = 0
            correct_action_idx = indices_used_for_data[0]
        else:
            indices_used_for_data = []
    else:
        indices_used_for_data = np.where(action_dataset.index.isin(applicable_actions_df.index))[0]
        
        if no_correct_idx is False:
            # Correct index
            applicable_actions_df = applicable_actions_df.loc[action_dataset.iloc[indices_used_for_data].index]
            correct_applicable_idx = (applicable_actions_df.index == row.name).argmax()
            correct_action_idx = indices_used_for_data[correct_applicable_idx]
    
    if no_correct_idx is True:
        return indices_used_for_data
    return indices_used_for_data, correct_applicable_idx, correct_action_idx


def get_ranking(pred, emb_for_comparison, correct_index, distance="euclidean", k=None):
    '''
    Get the rank of the prediction from the applicable actions.
    Returns (rank, [list_of_indices before <rank>])
    '''
    if distance == "euclidean":
        dist = ((emb_for_comparison-pred)**2).sum(axis=1)
    elif distance == "cosine":
        dist = 1 - torch.mm(emb_for_comparison, pred.view(-1, 1)).view(-1)/(torch.linalg.norm(emb_for_comparison, axis=1)*torch.linalg.norm(pred))

    # Get rank
    sorted_idx = dist.argsort()
    rank = (dist[sorted_idx] == dist[correct_index]).nonzero()[0] + 1
    list_of_indices = dist[sorted_idx[:rank]]
    
    # When the rank(correct_index) < k, then returns <rank, list>. So this extra condition - add some indices after rank(correct_index) to the list
    if k is not None:
        return sorted_idx[:k]
    return rank, list_of_indices

def get_top_k_indices(pred, emb_for_comparison, correct_index, distance="euclidean", k=1):
    return get_ranking(pred, emb_for_comparison, correct_index, distance, k)


class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.GIN = torch.load("models/zinc2m_gin.pth")
        self.DENSE = NeuralNet(self.GIN.output_dim*2, self.GIN.output_dim*2, num_hidden=3, hidden_size=256)
    
    @property
    def actor(self):
        return self.DENSE

    def forward(self, x1, x2, *args):
        out1 = self.GIN(x1, x1.node_feature.float())["graph_feature"]
        out2 = self.GIN(x2, x2.node_feature.float())["graph_feature"]
        
        out = torch.concatenate([out1, out2], axis=1)
        out = self.DENSE(out)
        return out
    
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.GIN = torch.load("models/zinc2m_gin.pth")
        self.DENSE = NeuralNet(self.GIN.output_dim*4, 1, num_hidden=2, hidden_size=256)
    
    def forward(self, x1, x2, x3, x4, *args):
        out1 = self.GIN(x1, x1.node_feature.float())["graph_feature"]
        out2 = self.GIN(x2, x2.node_feature.float())["graph_feature"]
        out3 = self.GIN(x3, x3.node_feature.float())["graph_feature"]
        out4 = self.GIN(x4, x4.node_feature.float())["graph_feature"]
        
        out = torch.concatenate([out1, out2, out3, out4], axis=1)
        out = self.DENSE(out)
        return out
    
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.GIN = torch.load("models/zinc2m_gin.pth")
        self.actor = NeuralNet(self.GIN.output_dim*2, self.GIN.output_dim*2, num_hidden=3, hidden_size=256)
        self.critic = NeuralNet(self.GIN.output_dim*4, 1, num_hidden=2, hidden_size=256)
    
    def forward(self, reac, prod, rsig, psig, out_type="both"):
        '''
        If out_type="actor", returns actions
        If out_type="critic", returns q_value
        If out_type="both", returns [actions, q_value]
        '''
        reac_out = self.GIN(reac, reac.node_feature.float())["graph_feature"]
        prod_out = self.GIN(prod, prod.node_feature.float())["graph_feature"]
    
        output = []
        if out_type in ["both", "actor"]:
            output.append(self.actor(torch.concatenate([reac_out, prod_out], axis=1)))

        if out_type in ["both", "critic"]:
            psig_out = self.GIN(psig, psig.node_feature.float())["graph_feature"]
            rsig_out = self.GIN(rsig, rsig.node_feature.float())["graph_feature"]
            output.append(self.critic(torch.concatenate([reac_out, prod_out, rsig_out, psig_out], axis=1)))
        
        if len(output) == 1:
            return output[0]
        return output
