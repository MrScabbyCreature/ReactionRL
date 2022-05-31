from rdkit import Chem
import pickle
import numpy as np
import pandas as pd
from IPython.display import display
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem

dataset = pd.read_csv("/home/abhor/Desktop/datasets/my_uspto/processed_data.csv", index_col=0)

# draw molecule with index
def mol_with_atom_index( mol ):
    mol = deepcopy(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def highlight_atoms(mol, hit_ats):
    '''
    Highlight the atoms in mol that have index in 'hit_ats'
    '''
#     # this is the code given in rdkit docs but doesn't actually work
#     d = rdMolDraw2D.MolDraw2DSVG(500, 500) # or MolDraw2DCairo to get PNGs
#     rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,)
    mol.__sssAtoms = hit_ats # workaround for now. Might not work in a later version of rdkit

class RLMol:
    def __init__(self, mol):
        self.mol = mol
    
    def display_mol(self, atom_num=False, highlight=False):
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(self.mol))
        if atom_num:
            mol = mol_with_atom_index(mol)
        if highlight:
            highlight_atoms(mol, self.sig)
        display(mol)
        
    def calculate_centres_and_signatures(self, common_subsequence, debug=False):
        # input
        mol = Chem.Mol(self.mol)
        cs = Chem.Mol(common_subsequence)
        
        # deal with atom indices
        mol_indices = list(range(mol.GetNumAtoms()))
        mol_indices_in_cs = np.array(rdchem.Mol(mol).GetSubstructMatch(cs))
        
        # find signature
        difference = list(set(mol_indices) - set(mol_indices_in_cs))
        self.sig = difference
        
        # find centre
        self.cen = []
        for idx in self.sig:
            atom = mol.GetAtomWithIdx(idx)
            neighbors = atom.GetNeighbors()
            neighbors_indices = list(map(lambda x: x.GetIdx(), neighbors))
            if set(neighbors_indices) - set(self.sig): # this atom has a neighbor outside of signature
                self.cen.append(idx)

        # if debug, display
        if debug:
            print("Signature")
            self.display_mol(atom_num=True, highlight=True)
            
            print("Centre at", self.cen)
    
    def get_signature(self):
        # calc Mol from list of ints
        sig = None
        mol = mol_with_atom_index(self.mol)
        with Chem.RWMol(mol) as mw:
            for idx in set(list(range(self.mol.GetNumAtoms()))) - set(self.sig):
                mw.RemoveAtom(idx)
            sig = Chem.Mol(mw)
        return mw
    
    def get_smiles_signature(self):
        return Chem.MolToSmiles(self.get_signature())
    
    def get_centre(self):
        return self.cen
                

class Reaction:
    def __init__(self, reactant, product):
        self.reactant = RLMol(reactant)
        self.product = RLMol(product)
        
    def _GetMCS(self):
        '''Get the Maximum common subsequence from reactant and product'''
        mcs = rdFMCS.FindMCS([self.reactant.mol, self.product.mol])
        return Chem.MolFromSmarts(mcs.smartsString)
    
    def display_reactant(self, atom_num=False, highlight=False):
        self.reactant.display_mol(atom_num, highlight)
            
            
    def display_product(self, atom_num=False, highlight=False):
        self.product.display_mol(atom_num, highlight)
    
    def calculate_centres_and_signatures(self, debug=False):
        '''
        Calculates centres and signatures from reactants and products
        Returns None
        '''
        mcs = self._GetMCS()
        if debug:
            print("Reactant\n")
        self.reactant.calculate_centres_and_signatures(mcs, debug)
        
        if debug:
            print("-"*100, "\nProduct\n")
        self.product.calculate_centres_and_signatures(mcs, debug)
    
    def get_signatures(self):
        # calc Mol from atom indices
        return self.reactant.get_signature(), self.product.get_signature()
    
    def get_smiles_signatures(self):
        return self.reactant.get_smiles_signature(), self.product.get_smiles_signature()
    
    def get_centres(self):
        # calc 
        return self.reactant.get_centre(), self.product.get_centre()
        
rsig_list = []
psig_list = []
rcen_list = []
pcen_list = []

for i in range(dataset.shape[0]):
    mol1 = Chem.MolFromSmiles(dataset["reactants"][i])
    mol2 = Chem.MolFromSmiles(dataset["products"][i])

    R = Reaction(mol1, mol2)
    R.calculate_centres_and_signatures()
    
    rcen, pcen = R.get_centres()
    rsig, psig = R.get_smiles_signatures()
    
    rsig_list.append(rsig)
    psig_list.append(psig)    
    rcen_list.append(rcen)    
    pcen_list.append(pcen)    
    
    print(i)

df = dataset.iloc[:100]

df = df.drop("reagents", axis=1)

df["rsig"] = rsig_list
df["psig"] = psig_list
df["rcen"] = rcen_list
df["pcen"] = pcen_list

df.to_csv("datasets/my_uspto/simulator_dataset.csv")