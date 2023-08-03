import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from mol_embedding.base_embedding_class import BaseEmbeddingClass
from torchdrug import data
import pandas as pd

class Zinc_GIN_Embedder(BaseEmbeddingClass):
    def __init__(self, model_path="models/zinc2m_gin.pth", device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.model = torch.load(model_path).to(self.device)
    
    def _torchdrug_mol(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        if mol.GetNumAtoms() == 1:
            mol = Chem.AddHs(mol)
        try:
            mol = data.Molecule.from_molecule(mol, atom_feature="pretrain", bond_feature="pretrain")
        except Exception:
            mol = data.Molecule.from_molecule(mol, atom_feature="pretrain", bond_feature="pretrain", with_hydrogen=True)
        return mol
    
    def mol_to_embedding(self, mol):
        # deepchem - attribute masking
        mol = self._torchdrug_mol(mol).to(self.device)
        emb = self.model(mol, mol.node_feature.float())["graph_feature"].reshape(-1)
        return emb.detach()

    def atom_to_embedding(self, mol, idx):
        mol = self._torchdrug_mol(mol).to(self.device)
        emb = self.model(mol, mol.node_feature.float())["node_feature"][idx]
        return emb.detach()
    
if __name__ == "__main__":
    GIN_embedder = Zinc_GIN_Embedder()
    print(GIN_embedder.mol_to_embedding(Chem.MolFromSmiles("CC")))
    print(GIN_embedder.atom_to_embedding(Chem.MolFromSmiles("CC"), 1))
    print(GIN_embedder.mol_to_embedding(Chem.MolFromSmiles("C")))
    print(GIN_embedder.atom_to_embedding(Chem.MolFromSmiles("C"), 0))
