import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem


# # Load processed data
dataset = pd.read_csv("datasets/processed_data.csv", index_col=0)

def get_connecting_atoms(mol, subgraph):
    '''
    Finds pair of atoms that connect 'mol' to 'subgraph'
    mol: Complete molecule
    subgraph: Part of mol (reaction signatire)
    
    Returns:
        [(A1, A2), (A1', A2'), ...]
        Where each tuple is a connected pair of atom such that A1 \in subgraph, A2 \in (mol-subgraph)
    '''
    # get subgraph matches
    matches = np.array(mol.GetSubstructMatches(subgraph))
    connections = []
    for match in matches:
        # atoms in connected subgraph
        atoms = [mol.GetAtomWithIdx(int(match[i])) for i in range(len(match))]
        for atom in atoms:
            # for each atom in connected subgraph, get neighbors
            neighbors = atom.GetNeighbors()
            neighbor_idx = set([n.GetIdx() for n in neighbors])
            # if there is a neighbor in (mol - subgraph), add connection to 
            if (neighbor_idx - set(match)):
                if len(neighbor_idx - set(match)) == 1:
                    connections.append((atom.GetIdx(), (neighbor_idx - set(match)).pop()))

    # remove repetitions
    connections = list(set(connections))
    return connections

def connect_substructures_if_applicable(mol, connections):
    '''
    If multiple connections have an atom in common, combine the substructures with the connection atom.
    Even if not, add the connecting atom to the substructure to maintain uniformity.
    mol: Molecule
    connections: (same as return of get_connecting_atoms)
    
    Returns:
        connections with substructures combined (same format as input connections)
    '''
    # combine if two connections have common atom in substructure
    d = {conn[1]: [] for conn in connections}
    for conn in connections:
        d[conn[1]].append(conn)
    connections = []
    for key in d:
        # combine
        atom = mol.GetAtomWithIdx(key)
        neighbors = atom.GetNeighbors()
        # get neighbors not in previous connections
        neighbor_idx = set([n.GetIdx() for n in neighbors]) - set([n[0] for n in d[key]])
        if len(neighbor_idx) == 1:
            for idx in neighbor_idx:
                connections.append((key, idx))
        else:
            # let it be
            connections.extend(d[key])
    return connections


# draw molecule with index
def mol_with_atom_index( mol ):
    mol = deepcopy(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def get_substructures(mol1, mol2):
    '''
    Takes 2 molecules. 
    Returns (common_substructure, reac_sig1, reac_sig2)
    '''
    # find common substruction (to later substract)
    res=rdFMCS.FindMCS([mol1, mol2])
    common_substructure = Chem.MolFromSmarts(res.smartsString)

    # reaction signature in mol 1
    mol1_diff = AllChem.DeleteSubstructs(mol1, common_substructure)

    # reaction signature in mol 2
    mol2_diff = AllChem.DeleteSubstructs(mol2, common_substructure)
    
    return common_substructure, mol1_diff, mol2_diff
    
def get_reaction_signatures(i):
    mol1 = Chem.MolFromSmiles(dataset["reactants"][i])
    mol2 = Chem.MolFromSmiles(dataset["products"][i])

    # get reaction signatures
    common_substructure, mol1_diff, mol2_diff = get_substructures(mol1, mol2)

    # mapping of molecule idx to common_substructure idx
    mol1_to_com_map = np.array(mol1.GetSubstructMatch(common_substructure))
    mol2_to_com_map = np.array(mol2.GetSubstructMatch(common_substructure))

    def comm_idx1(idx):
        '''
        return the atom idx in comm_sub corresponding to 'idx' in mol1
        '''
        arr = abs(np.array(mol1_to_com_map) - idx)
        if 0 not in arr:
            return -1
        return arr.argmin()

    def comm_idx2(idx):
        '''
        return the atom idx in comm_sub corresponding to 'idx' in mol2
        '''
        arr = abs(np.array(mol2_to_com_map) - idx)
        if 0 not in arr:
            return -1
        return arr.argmin()


    com_to_mol1_map = list(map(comm_idx1, mol1_to_com_map))
    com_to_mol2_map = list(map(comm_idx2, mol2_to_com_map))

    # get connection points
    con1 = get_connecting_atoms(mol1, mol1_diff)
    con2 = get_connecting_atoms(mol2, mol2_diff)

    con1 = list(filter(lambda x: (comm_idx1(x[1]) in com_to_mol1_map) and (comm_idx1(x[0]) not in com_to_mol1_map), con1))
    con2 = list(filter(lambda x: (comm_idx2(x[1]) in com_to_mol2_map) and (comm_idx2(x[0]) not in com_to_mol2_map), con2))
    
    con1, con2 = connect_substructures_if_applicable(mol1, con1), connect_substructures_if_applicable(mol2, con2)
    
    # verification
    assert len(con1) <= 1
    assert len(con2) <= 1
