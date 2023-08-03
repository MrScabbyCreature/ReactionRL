import deepchem as dc
import torch
from torch import nn
import dgl
import pickle
import sys
from mol_embedding.deepchem_featurizer import _featurize
from rdkit import Chem
from mol_embedding.base_embedding_class import BaseEmbeddingClass

dc.feat.MolGraphConvFeaturizer._featurize = _featurize

class Unpickler(pickle.Unpickler): # Pickle stores module info during torch.save so I have to allow it to search in a different module to torch.load
    def find_class(self, module, name):
        if module == "__main__":
            return super(Unpickler, self).find_class(__name__, name)
        return super(Unpickler, self).find_class(module, name)


class MPNNMolEmbedder(nn.Module):
    """MPNN embedder."""
    def __init__(self, gnn, readout):
        super(MPNNMolEmbedder, self).__init__()

        self.gnn = gnn
        self.readout = readout

    def _prepare_batch(self, g):
        dgl_graphs = [graph.to_dgl_graph() for graph in g]
        inputs = dgl.batch(dgl_graphs).to("cpu") # FIXME: Dynamic device(?)
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
        inputs = dgl.batch(dgl_graphs).to("cpu") # FIXME: Device????
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

class ChemBL_MPNN_Embedder(BaseEmbeddingClass):
    def __init__(self, mol_emb_model_path="models/MPNNMolEmbedder.pt", atom_emb_model_path="models/MPNNAtomEmbedder.pt"):
        super().__init__()
    
        # Featurizer
        self.f = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_partial_charge=True)

        # Model
        self.mol_em_model = torch.load(mol_emb_model_path, pickle_module=sys.modules[__name__])
        self.atom_em_model = torch.load(atom_emb_model_path, pickle_module=sys.modules[__name__])

    def mol_to_embedding(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        features = self.f.featurize([mol])[0]
        return self.mol_em_model([features])[0].cpu().detach().numpy()

    def atom_to_embedding(self, mol, idx):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        features = self.f.featurize([mol])[0]
        return self.atom_em_model([features], idx).cpu().detach().numpy()

if __name__ == "__main__":
    mpnn_embedder = ChemBL_MPNN_Embedder()
    print(mpnn_embedder.mol_to_embedding(Chem.MolFromSmiles("CC")))
    print(mpnn_embedder.atom_to_embedding(Chem.MolFromSmiles("CC"), 1))
    print(mpnn_embedder.mol_to_embedding(Chem.MolFromSmiles("C")))
    print(mpnn_embedder.atom_to_embedding(Chem.MolFromSmiles("C"), 0))
