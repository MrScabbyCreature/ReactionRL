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
dataset = dataset.iloc[:500]

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
        

def sig_and_cen_collector(df, return_dict = None):
    temp_rsig_list = []
    temp_psig_list = []
    temp_rcen_list = []
    temp_pcen_list = []

    for i in range(df.shape[0]):
        mol1 = Chem.MolFromSmiles(df["reactants"].iloc[i])
        mol2 = Chem.MolFromSmiles(df["products"].iloc[i])

        R = Reaction(mol1, mol2)
        R.calculate_centres_and_signatures()
        
        rcen, pcen = R.get_centres()
        rsig, psig = R.get_smiles_signatures()
        
        temp_rsig_list.append(rsig)
        temp_psig_list.append(psig)    
        temp_rcen_list.append(rcen)    
        temp_pcen_list.append(pcen)    

    if return_dict is not None:
        return_dict["rsig"].extend(temp_rsig_list)
        return_dict["psig"].extend(temp_psig_list)
        return_dict["rcen"].extend(temp_rcen_list)
        return_dict["pcen"].extend(temp_pcen_list)
    else:
        return temp_rsig_list, temp_psig_list, temp_rcen_list, temp_pcen_list

#######################################
# CALLING FUNCTIONS - MULTIPROCESSING #
#######################################

# multiprocess it - cuz some reactions go into infinite loops
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Manager
import time

def multiprocess_collector(df, return_dict):
    '''
    Collects signatures and centres for df using multiprocessing.
    Due to some reactions taking too long, the multiprocessing happens in recursion - 100 -> 10 -> 1 (size of df to process)
    Then all the results are collected and returned
    '''
    print(f"GOT DF OF SHAPE {df.shape}")
    
    for x in ["rsig", "psig", "rcen", "pcen"]:
        if x not in return_dict:
            return_dict[x] = []
    # spawn process to run on whole df
    p = Process(target=sig_and_cen_collector, args=(df, return_dict))
    p.start()
    start_time = time.time()

    # terminate if takes too long
    done = False
    kill_threshold = 10
    while not done:
        if p.is_alive():
            if time.time() - start_time > kill_threshold:
                p.kill()
                break
        else:
            done = True
    p.join()
    print(return_dict)

    # if completed successfully, return results
    if done:
        return return_dict["rsig"], return_dict["psig"], return_dict["rcen"], return_dict["pcen"]

    # not done - if df of size 1, return defaults instead
    if df.shape[0] == 1:
        return_dict["rsig"].append('')
        return_dict["psig"].append('')
        return_dict["rcen"].append([])
        return_dict["pcen"].append([])
        return [''], [''], [[]], [[]]
    
    # not done - else divide df into 10 parts and repeat
    elements = 100
    if df.shape[0] % 10 != 0:  # add some to make divisible by 10
        elements = df.shape[0]
        while df.shape[0] % 10 != 0:
            df = pd.concat([df, df.iloc[:100-df.shape[0]]])
    
    step_size = df.shape[0] // 10
    for i in range(10):
        multiprocess_collector(df.iloc[i*step_size: (i+1)*step_size], return_dict)
    return return_dict["rsig"][:elements], return_dict["psig"][:elements], return_dict["rcen"][:elements], return_dict["pcen"][:elements]



if __name__ == "__main__":
    # with ThreadPool(processes=1) as tp:
    #     tp.map()

    n = 100
    manager = Manager()

    rsig_list = []
    psig_list = []
    rcen_list = []
    pcen_list = []


    for i in range(dataset.shape[0]//n):
        print(i*n, min(i*n+n, dataset.shape[0]))
        man_dict = manager.dict()
        a, b, c, d = multiprocess_collector(dataset.iloc[i*n:min(i*n+n, dataset.shape[0])], man_dict)
        rsig_list.extend(a)
        psig_list.extend(b)
        rcen_list.extend(c)
        pcen_list.extend(d)

    print(rsig_list)

    dataset = dataset.drop("reagents", axis=1)

    dataset["rsig"] = rsig_list
    dataset["psig"] = psig_list
    dataset["rcen"] = rcen_list
    dataset["pcen"] = pcen_list

    dataset.to_csv("datasets/my_uspto/simulator_dataset.csv")