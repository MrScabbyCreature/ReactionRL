from rdkit import Chem
import time
import pickle
import pandas as pd
from IPython.display import display
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

start_mols = pickle.load(open("datasets/my_uspto/unique_start_mols.pickle", 'rb'))
categorical_probs_for_sampling_start_mols = None
device = None

def set_device(d):
    global device
    device = d
    return device

def _get_app_act_count(smile):
    return get_applicable_actions(Chem.MolFromSmiles(smile)).shape[0]


def calc_start_mol_prob_dist():
    global categorical_probs_for_sampling_start_mols
    # Probabilities for start mol sampling
    print("Calculating probability for start mol sampling")
    applicable_action_count = []
    with Pool(30) as p:
        for c in tqdm.tqdm(p.imap(_get_app_act_count, start_mols, chunksize=100), total=len(start_mols)):
            applicable_action_count.append(c)
    
    applicable_action_count = np.array(applicable_action_count)
    categorical_probs_for_sampling_start_mols = applicable_action_count / applicable_action_count.sum()


def _generate_train_data(smile, steps):
    mol = Chem.MolFromSmiles(smile)

    df = pd.DataFrame(columns=['reactant', 'rsub', 'rcen', 'rsig', 'rsig_cs_indices', 'psub', 'pcen', 'psig', 'psig_cs_indices', 'product'])
    index = []
    
    # Get sequences
    try:
        for i in range(steps):
            actions = get_applicable_actions(mol)
            if actions.shape[0] == 0:
                break

            # Apply a random action
            rand_idx = np.random.randint(0, actions.shape[0])
            product = apply_action(mol, *actions.iloc[rand_idx])

            # Add it to df
            df.loc[df.shape[0], :] = [Chem.MolToSmiles(mol)] + actions.iloc[rand_idx].tolist() + [Chem.MolToSmiles(product)]
            index.append(actions.iloc[rand_idx].name)

            # Next reactant = product
            mol = product
    except Exception as e:
        return pd.DataFrame(columns=['reactant', 'rsub', 'rcen', 'rsig', 'rsig_cs_indices', 'psub', 'pcen', 'psig', 'psig_cs_indices', 'product'])
    
    # Fix index
    df.index = index
    
    # Fix target
    df["product"] = Chem.MolToSmiles(product)

    return df

def generate_train_data(N, steps, multiprocess=True):
    # Generate dataset
    df_list = []
    final_shape = 0
    smiles_per_random_sample = 1000
    assert categorical_probs_for_sampling_start_mols is not None, "Run calc_start_mol_prob_dist() first to calc prob dist for start mols for sampling. This is done in multiprocess, so has to be run in __main__."

    # Create dataset for multi-step pred
    print("Creating dataset...")
    if multiprocess:
        with Pool(30) as p, tqdm.tqdm(total=N) as pbar:
            while final_shape < N:
                smiles = np.random.choice(start_mols, size=(smiles_per_random_sample,), p=categorical_probs_for_sampling_start_mols)

                for new_df in p.imap_unordered(functools.partial(_generate_train_data, steps=steps), smiles, chunksize=10):
                    df_list.append(new_df)
                    final_shape += new_df.shape[0]

                pbar.update(final_shape - pbar.n)
    else:
        raise Exception("Single process not implemented... :'(")

    main_df = pd.concat(df_list)
    return main_df



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

def train(X, Y, num_hidden=1, hidden_size=50, lr=1e-2, bs=64, epochs=100, batch_size=128):
    train_X = torch.Tensor(X[:int(X.shape[0]*0.8)]).to(device)
    train_Y = torch.Tensor(Y[:int(Y.shape[0]*0.8)]).to(device)

    test_X = torch.Tensor(X[int(X.shape[0]*0.8):]).to(device)
    test_Y = torch.Tensor(Y[int(Y.shape[0]*0.8):]).to(device)
    
    model = NeuralNet(train_X.shape[1], train_Y.shape[1], num_hidden=num_hidden, hidden_size=hidden_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

    loss_list = []
    test_loss = []

    # Train the model
    for epoch in range(epochs):
        for i in range(0, train_X.shape[0], batch_size):
            model.train()
            # Forward pass
            outputs = model(train_X[i:i+batch_size])
            loss = criterion(outputs, train_Y[i:i+batch_size])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())
        print ('Epoch {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
        
        model.eval()
        test_loss.append(criterion(model(test_X), test_Y).item()) 
    print("\nFINAL TEST LOSS:", test_loss[-1])
        
    plt.plot(loss_list[5:], label="training loss")
    plt.plot(test_loss[5:], label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
        
    return model


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
    mol = mol.to(device)
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
    # print(get_mol_embedding(model, rsub).shape)
    # print(get_atom_embedding(model, rsig, rcen).shape)
    # print(get_mol_embedding(model, rsig).shape)
    # print(get_mol_embedding(model, psub).shape)
    # print(get_atom_embedding(model, psig, pcen).shape)
    # print(get_mol_embedding(model, psig).shape)
    embedding = torch.concatenate([
                        # get_mol_embedding(model, rsub), 
                        # get_atom_embedding(model, rsig, rcen) / 5, 
                        get_mol_embedding(model, rsig), 
                        # get_mol_embedding(model, psub), 
                        # get_atom_embedding(model, psig, pcen) / 5, 
                        get_mol_embedding(model, psig)
                    ], axis=1)
    return embedding

def get_action_embedding_from_packed_molecule(model, rsig, psig):
    embedding = torch.concatenate([
                            get_mol_embedding(model, rsig), 
                        get_mol_embedding(model, psig)
                    ], axis=1)
    return embedding


def get_action_dataset_embeddings(model, action_rsigs, action_psigs):
    batch_size = 2048
    action_embeddings = []
    for i in tqdm.tqdm(range(0, action_dataset.shape[0], batch_size)):
        batch_rsig = action_rsigs[i:min(i+batch_size, action_dataset.shape[0])].to(device)
        batch_psig = action_psigs[i:min(i+batch_size, action_dataset.shape[0])].to(device)
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


# https://github.com/mangye16/ReID-Survey
def euclidean_dist(x, y):
    """
    Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
    Returns:
    dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    xy = x.matmul(y.t())

    m, n = x.size(0), y.size(0)
    xx = torch.linalg.norm(x, axis=1).expand(n, m).t()
    yy = torch.linalg.norm(y, axis=1).expand(m, n)
    
    return 1 - xy / (xx*yy)


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

class WeightedRegularizedTriplet(object):
    def __init__(self, dist="euclidean"):
        self.ranking_loss = nn.SoftMarginLoss()
        self.dist = dist

    def __call__(self, global_feat, labels):
        if self.dist=="euclidean":
            dist_mat = euclidean_dist(global_feat, global_feat)
        elif self.dist=="cosine":
            dist_mat = cosine_dist(global_feat, global_feat) ####### NEEEDS TO BE CHANGED!!!!!!!!!!!

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.GIN = torch.load("models/zinc2m_gin.pth")
        self.DENSE = torch.load("datasets/my_uspto/supervised_zinc_gin/mse_model.pth")
    
    def forward(self, x1, x2):
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
    
    def forward(self, x1, x2, x3, x4):
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
        self.actor = NeuralNet(self.GIN.output_dim*2, self.GIN.output_dim*2, num_hidden=3, hidden_size=500)
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
