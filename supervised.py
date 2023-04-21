from rdkit import Chem
import pickle, re
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import itertools
from tabulate import tabulate
from torchdrug import data
import argparse
from action_utils import *

import torch
import torch.nn as nn

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=100000, help="Number of data points to use")
parser.add_argument("--normalize", action="store_true", help="Whether to call standard normalize")
parser.add_argument("--hidden-size", type=int, default=512)
parser.add_argument("--num-hidden", type=int, default=3)
parser.add_argument("--lr", type=int, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--results-dir", type=str, default="results")
args = parser.parse_args()


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
    
#############
# Load data #
#############
main_df = pd.read_csv("datasets/my_uspto/supervised_zinc_gin/dataset.csv", index_col=0)
N = args.N

# take N samples of it
np.random.seed(42)
print(main_df.shape)
elements_to_fetch = np.random.randint(0, main_df.shape[0], size=(N,))
main_df = main_df.iloc[elements_to_fetch]
print(main_df.shape)

########################hidden_size, num_hidden, lr, epochs, batch_size = args.hidden_size, args.num_hidden, args.lr, args.epochs, args.batch_size

########################
model_name = "models/zinc2m_gin.pth"
clintox_gin = torch.load(model_name, map_location='cpu')


def clintox_gin_mol_embedding(smiles):
    # deepchem - attribute masking
    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        emb = clintox_gin(mol, mol.node_feature.float())["graph_feature"]
    except Exception as e:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        emb = clintox_gin(mol, mol.node_feature.float())["graph_feature"]
    return emb.detach().cpu()[0]

def clintox_gin_atom_embedding(smiles, idx):
    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        emb = clintox_gin(mol, mol.node_feature.float())["node_feature"][idx]
    except Exception as e:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        emb = clintox_gin(mol, mol.node_feature.float())["node_feature"][idx]
    return emb.detach().cpu()

def clinton_gin_action_embedding(action):
    rsub, rcen, rsig, _, psub, pcen, psig, __ = action
    embedding = np.concatenate([
#                         clintox_gin_mol_embedding(rsub), 
#                         clintox_gin_atom_embedding(rsig, rcen) / 5, 
                        clintox_gin_mol_embedding(rsig), 
#                         clintox_gin_mol_embedding(psub), 
#                         clintox_gin_atom_embedding(psig, pcen) / 5, 
                        clintox_gin_mol_embedding(psig)
                    ])
    return embedding

#####################
# Data for training #
#####################
X = []
for i in tqdm.tqdm(range(main_df.shape[0])):
    row = main_df.iloc[i]
    X.append(np.concatenate([clintox_gin_mol_embedding(row["reactant"]), clintox_gin_mol_embedding(row["product"])]))
X = np.stack(X)
emb_len = X.shape[1]//2
print("X.shape = ", X.shape)

Y = []
for i in tqdm.tqdm(range(main_df.shape[0])):
    Y.append(clinton_gin_action_embedding(main_df.iloc[i][main_df.columns[1:-1]]))
Y = np.stack(Y)
print("Y.shape", Y.shape)

#########################
# Get action embeddings #
#########################
action_dataset = pd.read_csv("datasets/my_uspto/action_dataset-filtered.csv", index_col=0)

action_dataset = action_dataset.loc[action_dataset["reactant_works"] & action_dataset["reactant_tested"] & action_dataset["action_tested"] & action_dataset["action_works"]]
action_dataset.shape

action_dataset = action_dataset[["rsub", "rcen", "rsig", "rbond", "psub", "pcen", "psig", "pbond"]]

print(action_dataset.shape)

action_embeddings = []
for i in tqdm.tqdm(range(action_dataset.shape[0])):
    action_embeddings.append(clinton_gin_action_embedding(action_dataset.iloc[i]))
action_embeddings = np.stack(action_embeddings)
print("action_embeddings.shape", action_embeddings.shape)

if args.normalize:
    from sklearn.preprocessing import StandardScaler

    std_scaler = StandardScaler()
    std_scaler.fit(X[:, 0:emb_len])

    for column_idx in range(0, X.shape[1], emb_len):
        X[:, column_idx:column_idx+emb_len] = std_scaler.transform(X[:, column_idx:column_idx+emb_len])

    for column_idx in range(0, Y.shape[1], emb_len):
        Y[:, column_idx:column_idx+emb_len] = std_scaler.transform(Y[:, column_idx:column_idx+emb_len])

    for column_idx in range(0, action_embeddings.shape[1], emb_len):
        action_embeddings[:, column_idx:column_idx+emb_len] = std_scaler.transform(action_embeddings[:, column_idx:column_idx+emb_len])


correct_indices = []
action_embedding_indices = []
for i in tqdm.tqdm(range(main_df.shape[0])):
    row = main_df.iloc[i]
    applicable_actions_df = get_applicable_actions(Chem.MolFromSmiles(row["reactant"]))
    
    indices_used_for_data = np.where(action_dataset.index.isin(applicable_actions_df.index))[0]
    action_embedding_indices.append(indices_used_for_data)

    applicable_actions_df = applicable_actions_df.loc[action_dataset.iloc[indices_used_for_data].index]
    correct_indices.append((applicable_actions_df.index == row.name).argmax())
    
    assert correct_indices[-1] < len(action_embedding_indices[-1]), f"WHAT!? {correct_indices[-1]} vs {len(indices_used_for_data)}"


def get_ranking(pred, emb_for_comparison, correct_index, distance="euclidean", k=None):
    '''
    Get the rank of the prediction from the applicable actions.
    Returns (rank, [list_of_indices before <rank>])
    '''
    if distance == "euclidean":
        dist = ((emb_for_comparison-pred)**2).sum(axis=1)
    elif distance == "cosine":
        dist = 1 - (emb_for_comparison.dot(pred))/(np.linalg.norm(emb_for_comparison, axis=1)*np.linalg.norm(pred))

    maxy = max(dist)

    list_of_indices = []
    for attempt in range(dist.shape[0]):
        miny = dist.argmin()
        # print(miny, correct_index, dist[correct_index], min(dist), maxy)
        if dist[miny] == dist[correct_index]:
            # print(i, attempt)
            break
        else:
            list_of_indices.append(miny)
            if k is not None and len(list_of_indices) == k:
                return list_of_indices
            dist[miny] = 100000
    
    # When the rank(correct_index) < k, then returns <rank, list>. So this extra condition - add some indices after rank(correct_index) to the list
    if k is not None:
        dist[miny] = 100000
        for attempt in range(min(k, emb_for_comparison.shape[0]-1) - len(list_of_indices)):
            miny = dist.argmin()
            list_of_indices.append(miny)
            dist[miny] = 100000
        return list_of_indices
    return attempt, list_of_indices

def get_top_k_indices(pred, emb_for_comparison, correct_index, distance="euclidean", k=1):
    return get_ranking(pred, emb_for_comparison, correct_index, distance, k)
    

# Main training
train_idx = np.arange(0, int(args.N*0.8))
test_idx = np.arange(int(args.N*0.8), N)

train_X = torch.Tensor(X[train_idx])
train_Y = torch.Tensor(Y[train_idx])

test_X = torch.Tensor(X[test_idx])
test_Y = torch.Tensor(Y[test_idx])

hidden_size, num_hidden, lr, epochs, batch_size = args.hidden_size, args.num_hidden, args.lr, args.epochs, args.batch_size
print("hidden_size, num_hidden, lr, epochs, batch_size =", hidden_size, num_hidden, lr, epochs, batch_size)

model = NeuralNet(train_X.shape[1], train_Y.shape[1], num_hidden=num_hidden, hidden_size=hidden_size)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

for margin, distance_metric in itertools.product([10000], ["euclidean", "cosine"]):
    print("@"*190)
    print("@"*190)
    print("@"*190)
    if distance_metric == "euclidean":
        criterion = nn.TripletMarginLoss(margin=margin)
    elif distance_metric == "cosine":
        criterion = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y))

    loss_list = []
    test_loss = []

    results_dict = {}
    results_dict["rmse"] = []
    results_dict["cosine_sim"] = []
    results_dict["Rank(euclidean)"] = []
    results_dict["Rank(cosine)"] = []

    for epoch in range(epochs):
        ############
        # Training #
        ############
        model.train()
        for i in range(0, train_X.shape[0], batch_size):
            # Forward pass
            outputs = model(train_X[i:i+batch_size])

            # Calc negatives
            positive_index = []
            negatives = []
            for _i in range(outputs.shape[0]):
                act_emb_for_i, correct_index = action_embeddings[action_embedding_indices[train_idx[i+_i]]], correct_indices[train_idx[i+_i]]
                top = get_top_k_indices(outputs[_i].detach().cpu().numpy(), act_emb_for_i, correct_index, distance=distance_metric, k=50)

                positive_index.extend([_i]*len(top))
                negatives.append(act_emb_for_i[top])
            negatives = np.concatenate(negatives, axis=0)
            
            # Calc loss
            loss = criterion(outputs[positive_index], train_Y[i:i+batch_size][positive_index], torch.Tensor(negatives))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        print ('{:.6f}({})'.format(loss.item(), epoch+1), end='  ')
        
        # SWITCH INDENT HERE ----
        model.eval()
        with torch.no_grad():
            print()

            margin_string = f"# Margin = {margin} | distance_metric = {distance_metric} | topk distance = {distance_metric} #"
            print("#" * len(margin_string))
            print(margin_string)
            print("#" * len(margin_string))

            # Predictions and action component-wise loss
            pred = model(test_X).detach().numpy() 
            true = test_Y.detach().numpy()

            metric_df = pd.DataFrame(columns=["rMSE", "Cosine", "Rank(euclid)", "Rank(cosine)"])

            # Print Test metrics
            results_dict["rmse"].append( (((pred-true)**2).sum(axis=1)**0.5).mean() )
            results_dict["cosine_sim"].append(((pred*true).sum(axis=1) / np.linalg.norm(pred, axis=1) / np.linalg.norm(true, axis=1)).mean())    

            # Print Test metric - Rank
            for dist in ["euclidean", "cosine"]:
                rank_list = []
                l = []
                total = []
                for i in range(pred.shape[0]):
                    pred_for_i = pred[i]
                    act_emb_for_i, correct_index = action_embeddings[action_embedding_indices[test_idx[i]]], correct_indices[test_idx[i]]

                    rank, list_of_indices = get_ranking(pred_for_i, act_emb_for_i, correct_index, distance=dist)
                    l.append(rank)
                    total.append(act_emb_for_i.shape[0])
                results_dict[f"Rank({dist})"].append(f"{np.mean(l):.4f}({np.mean(total)}) +- {np.std(l):.4f}")

            metric_df["rMSE"] = [results_dict["rmse"][-1]]
            metric_df["Cosine"] = [results_dict["cosine_sim"][-1]]
            metric_df["Rank(euclid)"] = [results_dict["Rank(euclidean)"][-1]]
            metric_df["Rank(cosine)"] = [results_dict["Rank(cosine)"][-1]]
            print(tabulate(metric_df, headers='keys', tablefmt='fancy_grid'))
            print()
    
    # Dump results
    results_file_name = re.sub("_results_dir=[^_]*", "", "_".join(list(map(lambda x: "=".join(list(map(str, x))), vars(args).items())))) # args
    results_file_name += f"_m={margin}"
    results_file_name += f"_dist={distance_metric}"
    results_file_name += ".pickle"
    results_file_name = os.path.join(args.results_dir, results_file_name)
    
    pickle.dump(results_dict, open(results_file_name, 'wb'))
    print("Dumped file at", results_file_name)