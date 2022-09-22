import deepchem as dc
import torch
from torch import nn
import dgl

class MPNNMolEmbedder(nn.Module):
    """MPNN embedder."""
    def __init__(self, gnn, readout):
        super(MPNNMolEmbedder, self).__init__()

        self.gnn = gnn
        self.readout = readout

    def _prepare_batch(self, g):
        dgl_graphs = [graph.to_dgl_graph() for graph in g]
        inputs = dgl.batch(dgl_graphs).to("cpu")
        return inputs
        
    def forward(self, g):
        """
        Parameters
        ----------
        g : GraphData
            GraphData for a batch of graphs.

        Returns
        -------
        graph embeddings
        """
        dgl_g = self._prepare_batch(g)
        node_feats = self.gnn(dgl_g, dgl_g.ndata["x"], dgl_g.edata["edge_attr"])
        graph_feats = self.readout(dgl_g, node_feats)
        return graph_feats

class MPNNAtomEmbedder(nn.Module):
    """MPNN embedder."""
    def __init__(self, gnn):
        super(MPNNAtomEmbedder, self).__init__()
        self.gnn = gnn

    def _prepare_batch(self, g):
        dgl_graphs = [graph.to_dgl_graph() for graph in g]
        inputs = dgl.batch(dgl_graphs).to("cpu")
        return inputs
        
    def forward(self, g, idx):
        """
        Parameters
        ----------
        g : GraphData
            GraphData for a batch of graphs.

        Returns
        -------
        graph embeddings
        """
        dgl_g = self._prepare_batch(g)
        node_feats = self.gnn(dgl_g, dgl_g.ndata["x"], dgl_g.edata["edge_attr"])
        return node_feats[idx]

# Featurizer
f = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_partial_charge=True)

# Model
mol_em_model = torch.load("models/MPNNMolEmbedder.pt")
atom_em_model = torch.load("models/MPNNAtomEmbedder.pt")

def mol_to_embedding(mol):
    features = f.featurize([mol])[0]
    return mol_em_model([features])[0]

def atom_to_embedding(mol, idx):
    features = f.featurize([mol])[0]
    return atom_em_model([features], idx)

if __name__ == "__main__":
    from rdkit import Chem
    print(mol_to_embedding(Chem.MolFromSmiles("CCCC")).shape)
    print(atom_to_embedding(Chem.MolFromSmiles("CCCC"), 3).shape)