'''
This code implements functions that the gym environment uses for molecule based actions.
'''
from rdkit import Chem
import numpy as np
import pandas as pd
from utils import get_mol_certificate
import networkx as nx
import tqdm
import pickle

dataset = pd.read_csv("/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/action_dataset.csv", index_col=0)
path_to_rsig_cluster_dict = "/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/rsig_cluster_dict.pickle"
path_to_certi_dict = "/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/certi_dict.pickle"

# Fetch the rsigs using the clusters
try:
    rsig_cluster_to_rsig_d = pickle.load(open(path_to_rsig_cluster_dict, 'rb'))
except Exception as e: 
    print("Calculating rsig_cluster dict....")
    rsig_cluster_to_rsig_d = {}
    for cluster_id in tqdm.tqdm(dataset["rsig_clusters"].unique()):
        cluster_df = dataset[dataset["rsig_clusters"] == cluster_id]
        rsig  = Chem.MolFromSmiles(cluster_df.iloc[0]["rsig"])
        rsig_cluster_to_rsig_d[cluster_id] = rsig
    pickle.dump(rsig_cluster_to_rsig_d, open(path_to_rsig_cluster_dict, 'wb'))
    
# Make a mapping of certificates and cluster_ids
try:
    certificate_to_cluster_id_dict = pickle.load(open(path_to_certi_dict, 'rb'))
except Exception as e: 
    print("Calculating certificate dict....")
    certificate_to_cluster_id_dict = {}
    for _id in tqdm.tqdm(rsig_cluster_to_rsig_d):
        C = get_mol_certificate(rsig_cluster_to_rsig_d[_id])
        if C in certificate_to_cluster_id_dict:
            certificate_to_cluster_id_dict[C].append(_id)
        else:
            certificate_to_cluster_id_dict[C] = [_id]
    pickle.dump(certificate_to_cluster_id_dict, open(path_to_certi_dict, 'wb'))

def add_immediate_neighbors(mol, indices):
    '''
    Add immediate neighbors of 'indices' in 'mol' to 'indices'.
    '''
    def _add_neighbors(idx_list):
        atoms = list(map(lambda x: mol.GetAtomWithIdx(int(x)), idx_list))
        neighbors = []
        for atom in atoms:
            neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
        return np.unique(neighbors).tolist()
    
    # first add immediate neighbors
    indices = _add_neighbors(indices)
    
    # if any aromtic atoms in neighbors, add them as well
    repeat = True
    while repeat:
        repeat = False
        for n in set(_add_neighbors(indices)) - set(indices):
            if mol.GetAtomWithIdx(int(n)).GetIsAromatic():
                indices.append(n)
                repeat = True
    
    return np.unique(indices)

def verify_action_applicability(mol, r_indices, cluster_id):
    '''
    Veriy action applicability on 'r_indices' of 'cluster_id'

    returns True or False
    '''
    mol = Chem.Mol(mol)
    rsig = Chem.MolFromSmiles(dataset[dataset["rsig_clusters"]==cluster_id].iloc[0]["rsig"])
    rsub = Chem.MolFromSmiles(dataset[dataset["rsig_clusters"]==cluster_id].iloc[0]["rsub"])
        
    # Get the correct rsig_match
    rsig_matches = mol.GetSubstructMatches(rsig)
    if not rsig_matches:
        rsig_match = ()
    else:
        for rsig_match in rsig_matches:
            if not (set(rsig_match) - set(r_indices)):
                break
    
    # Get the correct rsub_match(s)
    temp_rsub_matches = mol.GetSubstructMatches(rsub)
    rsub_matches = []    
    for rsub_match in temp_rsub_matches:
        if not (set(rsub_match) - set(r_indices)):
            rsub_matches.append(rsub_match)
    
    # is there is a rsub whose neighbors are in rsig_match, we're good
    result = False
    for rsub_match in rsub_matches:
        # get neighbors
        atoms = list(map(lambda x: mol.GetAtomWithIdx(x), rsub_match))
        neighbors = []
        for atom in atoms:
            neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
        neighbors = np.unique(neighbors)
        if not set(neighbors) - set(rsig_match):
            result = True

    return result
    
def get_mol_from_index_list(mol, indices):
    '''
    Get the sub-mol from 'mol' using 'indices'
    '''
    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for idx in set(list(range(mol.GetNumAtoms()))) - set(indices):
        rw.RemoveAtom(idx)
    rw.CommitBatchEdit()
    return Chem.Mol(rw)
    
def get_applicable_rsig_clusters(in_mol):
    # For each cut vertex, we find two disconnected components and search the smaller one in our index
    G = nx.from_numpy_matrix(Chem.GetAdjacencyMatrix(in_mol))
    applicable_clusters = []

    for x in nx.articulation_points(G):
        # Remove atom (not directly, otherwise the index resets)
        # First remove bonds to x
        mw = Chem.RWMol(in_mol)
        Chem.Kekulize(mw, clearAromaticFlags=True)
        for n in mw.GetAtomWithIdx(x).GetNeighbors():
            mw.RemoveBond(x, n.GetIdx())

        # Find fragments
        mol_frags = list(Chem.rdmolops.GetMolFrags(mw))

        # Remove x from fragments
        mol_frags.remove((x,))

        # For each fragment except the biggest, add x and extract sub-molecule and search
        for frag in sorted(mol_frags, key=lambda x: len(x))[:-1]:
            indices = [x] + list(frag)

            for _ in range(2):
                # we add neighbors twice to rsub and then search for rsig
                indices = add_immediate_neighbors(in_mol, indices)
                candidate = get_mol_from_index_list(in_mol, indices)
                try:
                    Chem.SanitizeMol(candidate)
                except Exception as e:
                    pass

                # get certificate and search in rsig
                cand_certi = get_mol_certificate(candidate)

                if cand_certi in certificate_to_cluster_id_dict:
                    # Verify rsig
                    for cluster_id in certificate_to_cluster_id_dict[cand_certi]:
                        if verify_action_applicability(in_mol, indices, cluster_id):
                            if cluster_id not in applicable_clusters:
                                applicable_clusters.append(cluster_id)
    return applicable_clusters

def get_random_action(mol):
    applicable_clusters = get_applicable_rsig_clusters(mol)
    return_format = ["rsig", "rsub", "rcen", "psig", "psub", "pcen"]
    return dataset[dataset["rsig_clusters"].isin(applicable_clusters)].sample()[return_format]

from rdkit.Chem import rdFMCS

def mol_with_atom_index( mol ):
    '''
    draw molecule with index
    '''
    colored = False
    if hasattr(mol, "__sssAtoms"):
        sss = mol.__sssAtoms
        colored = True
    mol = Chem.Mol(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    if colored:
        mol.__sssAtoms = sss
    return mol

def smiles_without_atom_index( smiles ):
    '''
    Convert smiles with numbers to smiles without numbers
    '''
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).ClearProp( 'molAtomMapNumber' )
    return Chem.MolToSmiles(mol)

def find_connecting_atoms_not_in_sig(mol, sig_indices, centre):
    cen_atom = mol.GetAtomWithIdx(centre)
    neighbors_indices = list(map(lambda x: x.GetIdx(), cen_atom.GetNeighbors()))
    return set(neighbors_indices) - set(neighbors_indices)


def apply_action(input_mol, rsig, rsub, rcen, psig, psub, pcen):
    input_mol = Chem.Mol(input_mol)
    rsig = Chem.MolFromSmiles(rsig)
    psig = Chem.MolFromSmiles(psig)
    rsig_matches = input_mol.GetSubstructMatches(rsig)
    # FIXME: Provide option to use more than just the first match
    rsig_match = rsig_matches[0]
#     print(rsig_match)

    # Replace rsig with psig
    mcs = rdFMCS.FindMCS([rsig, psig])
    cs = Chem.MolFromSmarts(mcs.smartsString)
    rsig_cs_indices = Chem.Mol(rsig).GetSubstructMatch(cs)
    psig_cs_indices = Chem.Mol(psig).GetSubstructMatch(cs)
    
    rwmol = Chem.RWMol(input_mol)
    rwmol.InsertMol(Chem.Mol(psig))
    for r_idx, p_idx in zip(rsig_cs_indices, psig_cs_indices):
        for conn in find_connecting_atoms_not_in_sig(in_mol, rsig_match, r_idx):
            rwmol.AddBond(p_idx, conn, input_mol.GetBondBetweenAtoms(r_idx, conn))
            rwmol.RemoveBond(r_idx, conn)

    
