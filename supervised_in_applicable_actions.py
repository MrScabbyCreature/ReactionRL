from rdkit import Chem
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
import tqdm
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool
from torchdrug import data
import glob

from action_utils import get_applicable_actions

import warnings
warnings.filterwarnings("ignore")

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
    

def train(X, Y, num_hidden=1, hidden_size=50, lr=1e-2, batch_size=64, epochs=100):
    train_X = torch.Tensor(X[:int(X.shape[0]*0.7)])
    train_Y = torch.Tensor(Y[:int(Y.shape[0]*0.7)])

    test_X = torch.Tensor(X[int(X.shape[0]*0.7):])
    test_Y = torch.Tensor(Y[int(Y.shape[0]*0.7):])
    
    model = NeuralNet(train_X.shape[1], train_Y.shape[1], num_hidden=num_hidden, hidden_size=hidden_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

    loss_list = []
    test_loss = []

    # Train the model
    for epoch in range(epochs):
        for i in range(0, train_X.shape[0], batch_size):
            # Forward pass
            outputs = model(train_X[i:i+batch_size])
            loss = criterion(outputs, train_Y[i:i+batch_size])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())
        print ('Epoch {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
        
        test_loss.append(criterion(model(test_X), test_Y).item()) 
    print("\nFINAL TEST LOSS:", test_loss[-1])
        
    plt.plot(loss_list, label="training loss")
    plt.plot(test_loss, label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
        
    return model


def zinc_gin_mol_embedding(smiles):
    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        emb = zinc_gin(mol, mol.node_feature.float())["graph_feature"]
    except Exception as e:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        emb = zinc_gin(mol, mol.node_feature.float())["graph_feature"]
    return emb.detach().cpu()[0]

def zinc_gin_atom_embedding(smiles, idx):
    try:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain")
        emb = zinc_gin(mol, mol.node_feature.float())["node_feature"][idx]
    except Exception as e:
        mol = data.Molecule.from_smiles(smiles, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        emb = zinc_gin(mol, mol.node_feature.float())["node_feature"][idx]
    return emb.detach().cpu()

def zinc_gin_action_embedding(action):
    rsub, rcen, rsig, _, psub, pcen, psig, __ = action
    embedding = np.concatenate([
#                         zinc_gin_mol_embedding(rsub), 
                        zinc_gin_atom_embedding(rsig, rcen) / 5, 
                        zinc_gin_mol_embedding(rsig), 
#                         zinc_gin_mol_embedding(psub), 
                        zinc_gin_atom_embedding(psig, pcen) / 5, 
                        zinc_gin_mol_embedding(psig)
                    ])
    return embedding

def get_pred_index(args):
    correct_index, dist = args

    maxy = max(dist)

    for attempt in range(action_dataset.shape[0]):
        miny = dist.argmin()
        # print(miny, correct_index, dist[correct_index], min(dist), maxy)
        if dist[miny] == dist[correct_index]:
            # print(i, attempt)
            break
        dist[miny] = 100000
    return attempt

if __name__ == "__main__":
    # Load dataset for training the NN 
    main_df = pd.read_csv("datasets/my_uspto/supervised_zinc_gin/dataset.csv", index_col=0)
    main_df.shape

    # Load action dataset
    action_dataset = pd.read_csv("datasets/my_uspto/action_dataset-filtered.csv", index_col=0)
    action_dataset = action_dataset.loc[action_dataset["reactant_works"] & action_dataset["reactant_tested"] & action_dataset["action_tested"] & action_dataset["action_works"]]
    action_dataset.shape
    action_dataset = action_dataset[["rsub", "rcen", "rsig", "rbond", "psub", "pcen", "psig", "pbond"]]
    action_dataset.shape

    # Get model list for zinc dataset
    model_list = glob.glob("models/zinc*")

    # Main loop - It does it all~~
    for model_path in model_list:
        print("#"*100)

        ################
        # Load a model #
        ################
        print(f"Loading model \033[93m{model_path}\033[0m")
        zinc_gin = torch.load(model_path)

        ##############################
        # Create train and test data #
        ##############################
        X = np.stack(main_df.apply(lambda x: np.concatenate([zinc_gin_mol_embedding(x["reactant"]), zinc_gin_mol_embedding(x["product"])]), axis=1).tolist())
        print("X shape:", X.shape)

        Y = np.stack([zinc_gin_action_embedding(main_df.iloc[i][main_df.columns[1:-1]]) for i in range(main_df.shape[0])])
        print("Y shape:", Y.shape)

        # Target = target - source (gives better results)
        emb_len = X.shape[1]//2
        X[:, :emb_len] -= X[:, emb_len:] # This makes predictions better (target = target - source)

        #################
        # Train a model #
        #################
        model = train(X, Y, hidden_size=500, num_hidden=2, lr=1e-3, epochs=20)

        ################################################
        # get prediction socres for each sub-embedding #
        ################################################
        pred = model(torch.Tensor(X[10000:])).detach().numpy() 
        true = Y[10000:]

        l = emb_len*np.arange(5)

        print("Sub-embedding losses:")
        for i in range(len(l)-1):
            print((((pred[:, l[i]:l[i+1]] - true[:, l[i]:l[i+1]]))**2).sum()/3000 / (l[i+1]-l[i]))
        print()

        #############################
        # Compute Action embeddings #
        #############################
        action_embeddings = np.stack([zinc_gin_action_embedding(action_dataset.iloc[i])] for i in range(action_dataset.shape[0]))

        #################
        # Print results #
        #################
        P = Pool(10)
        string = f"\033[94m\033[1m{model_path} ---- \033[0m\033[0m"
        for idx in emb_len*np.arange(4):
            l = []
            args = []
            # collect args
            for i in range(10000, 13000):
                applicable_actions_df = get_applicable_actions(Chem.MolFromSmiles(main_df["reactant"].iloc[i]))
                assert main_df.iloc[i].name in applicable_actions_df.index, f"The chosen action is not in applicable actions??? i = {i}"
                args.append(
                        (
                            (applicable_actions_df.index == main_df.iloc[i].name).argmax(), 
                            ((action_embeddings[action_dataset.index.isin(applicable_actions_df.index)] - pred[i])[:, idx:idx+emb_len]**2).sum(axis=1)
                    )
                )
            
            for item in tqdm.tqdm(P.imap_unordered(get_pred_index, args, chunksize=5)):
                l.append(item)
            
            string += f"\n\033[94m\033[1m{idx}:{idx+emb_len} = {np.mean(l)} +- {np.std(l)}\033[0m\033[0m"
        print()
        print(string)