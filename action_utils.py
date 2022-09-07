'''
This code implements functions that the gym environment uses for molecule based actions.
'''
from rdkit import Chem
import numpy as np
import pandas as pd
from utils import *
import tqdm
import pickle
import networkx as nx

dataset = pd.read_csv("/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/action_dataset-filtered.csv", index_col=0)
dataset = dataset[dataset["action_works"] & dataset["action_tested"]]
path_to_rsig_cluster_dict = "/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/rsig_cluster_dict.pickle"
path_to_certi_dict = "/home/abhor/Desktop/repos/ReactionRL/datasets/my_uspto/certi_dict.pickle"

# Fetch the rsigs using the clusters # TODO: Dump the pickles if the csv is updated(md5?)
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

def add_immediate_neighbors(mol, indices, add_aromatic_cycles=True):
    def _add_neighbors(idx_list):
        atoms = list(map(lambda x: mol.GetAtomWithIdx(int(x)), idx_list))
        neighbors = []
        for atom in atoms:
            neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
        return np.unique(neighbors).tolist()
    
    # first add immediate neighbors
    new_indices = _add_neighbors(indices)
    
    if set(new_indices) == set(indices): # indices = whole molecule
        return indices
    
    # if added neighbor is aromatic, we have to check for more neighbors (if I do not add this condition, it does neighbor+1)
    if add_aromatic_cycles and any(list(map(lambda idx: mol.GetAtomWithIdx(idx).GetIsAromatic(), list(set(new_indices) - set(indices))))):
        indices = list(new_indices)
        # if any aromtic atoms in neighbors, add them as well
        repeat = True
        while repeat:
            repeat = False
            for n in set(_add_neighbors(indices)) - set(indices):
                if mol.GetAtomWithIdx(int(n)).GetIsAromatic():
                    indices.append(n)
                    repeat = True
    else:
        indices = new_indices
    
    return np.unique(indices)

def verify_action_applicability(mol, r_indices, cluster_id):
    mol = Chem.Mol(mol)
    rsig = Chem.MolFromSmiles(dataset[dataset["rsig_clusters"]==cluster_id].iloc[0]["rsig"])
    rsub = Chem.MolFromSmiles(dataset[dataset["rsig_clusters"]==cluster_id].iloc[0]["rsub"])
    rcen = dataset[dataset["rsig_clusters"]==cluster_id].iloc[0]["rcen"]
    rbond = dataset[dataset["rsig_clusters"]==cluster_id].iloc[0]["rbond"]
    rbond = list(map(float, rbond.replace("[", "").replace("]", "").replace(" ", "").split(",")))
            
    # Get the correct rsig_match
    rsig_matches = mol.GetSubstructMatches(rsig)
    if not rsig_matches:
        rsig_match = ()
    else:
        for rsig_match in rsig_matches:
            if not (set(rsig_match) - set(r_indices)):
                break
    
    atm_map_nums = []
    for i in range(rsub.GetNumAtoms()):
        atm_map_nums.append(rsub.GetAtomWithIdx(i).GetAtomMapNum())
        
    try:
        rsub_match = np.array(rsig_match)[atm_map_nums].tolist() # no rsig match found
    except Exception as e:
        return False

    # get neighbors
    atoms = list(map(lambda x: mol.GetAtomWithIdx(x), rsub_match))
    neighbors = []
    for atom in atoms:
        neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
    neighbors = np.unique(neighbors)

    if not set(neighbors) - set(rsig_match):
        return True
    return False
    
def get_mol_from_index_list(mol, indices):
    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for idx in set(list(range(mol.GetNumAtoms()))) - set(indices):
        rw.RemoveAtom(idx)
    rw.CommitBatchEdit()
    return Chem.Mol(rw)


    
def get_applicable_rsig_clusters(in_mol):
    # For each cut vertex, we find two disconnected components and search the smaller one in our index
    G = nx.from_numpy_array(Chem.GetAdjacencyMatrix(in_mol))
    applicable_actions = []

    for x in nx.articulation_points(G):
        # Remove atom (not directly, otherwise the index resets)
        # First remove bonds to x
        in_mol_kekulized = Chem.Mol(in_mol)
        Chem.Kekulize(in_mol_kekulized, clearAromaticFlags=True)
        mw = Chem.RWMol(in_mol_kekulized)
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
                candidate = get_mol_from_index_list(in_mol_kekulized, indices)
                try:
                    Chem.SanitizeMol(candidate)
                except Exception as e:
                    print("ERORORORORORORORRO")
                    print(e)

                # get certificate and search in rsig
                cand_certi = get_mol_certificate(candidate)

                if cand_certi in certificate_to_cluster_id_dict:
                    # Verify rsig
                    for cluster_id in certificate_to_cluster_id_dict[cand_certi]:
                        if verify_action_applicability(in_mol, indices, cluster_id):
                            if cluster_id not in applicable_actions:
                                applicable_actions.append(cluster_id)
    return applicable_actions

def get_random_action(mol, random_state=40):
    applicable_clusters = get_applicable_rsig_clusters(mol)
    return_format = ["rsub", "rcen", "rsig", "rsig_cs_indices", "psub", "pcen", "psig", "psig_cs_indices"]

    # random sample
    sample = dataset[dataset["rsig_clusters"].isin(applicable_clusters)].sample(random_state=random_state)[return_format].iloc[0]
    print("Action loc", sample.name)
    return sample.values

def filter_sensible_rsig_matches(mol, rsig_matches, rsig, rsub, rcen):
    '''
    Checks if the rsub in given rsig only has neighbors within rsig (if there are neighbors outside rsig, then this is not a valid rsub by definition of its creation (rsig = rsub + 2 neighbors))
    '''
    # Get the atoms in rsig corresponding to rsub
    rsub_atom_indices = []
    for atom in rsub.GetAtoms():
        rsub_atom_indices.append(atom.GetAtomMapNum())#GetAtomWithIdx( idx ).GetProp( 'molAtomMapNumber'))
    rsig_atom_indices_in_rsub = list(map(lambda x: GetAtomWithAtomMapNum(rsig, x).GetIdx(), rsub_atom_indices))
    
    # Corresponding atoms in mol should have neighbors inside rsig_matches
    def verify(match):
        neighbors = []
        for idx in rsig_atom_indices_in_rsub:
            corr_idx = match[idx]
            atom = mol.GetAtomWithIdx(corr_idx)
            neighbors.extend(list(map(lambda x: x.GetIdx(), atom.GetNeighbors())))
        if not set(neighbors) - set(match):
            return True
        return False
        
    rsig_matches = list(filter(verify, rsig_matches))
    return rsig_matches

def apply_action(input_mol, rsub, rcen, rsig, rsig_cs_indices, psub, pcen, psig, psig_cs_indices):
    # Some basic conversions acc. to dataset format
    input_mol = Chem.Mol(input_mol)
    rsig = Chem.MolFromSmiles(rsig)
    psig = Chem.MolFromSmiles(psig)
    rsig_cs_indices = list(map(int, rsig_cs_indices.split(".")))
    psig_cs_indices = list(map(int, psig_cs_indices.split(".")))
    
    # Find rsig in input_mol
    rsig_matches = input_mol.GetSubstructMatches(rsig)

    # If multiple matches, choose one where rsub/rcen makes sense
    if len(rsig_matches) > 1:
        rsig_matches = filter_sensible_rsig_matches(input_mol, rsig_matches, rsig, Chem.MolFromSmiles(rsub), rcen)
    
    # FIXME: Provide option to use more than just the first match
    rsig_match = rsig_matches[0]
    
    # Find indices to be exchanged
    input_mol_cs_indices = np.array(rsig_match)[rsig_cs_indices].tolist()
    
    # Exchange indices (replace bonds for atoms at given indices)
    rwmol = Chem.RWMol(mol_with_atom_index(input_mol))
    num_atoms = input_mol.GetNumAtoms()
    new_psig = Chem.Mol(psig)
    for atom in new_psig.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum() + num_atoms)
    rwmol.InsertMol(Chem.Mol(new_psig))
    
    # TODO: FIND A BETTER WAY TO FIND THE INDICES OF THE TWO CONNECTED COMPONENTS
    rsig_cs_atom_map_num = list(input_mol_cs_indices)
    psig_cs_atom_map_num = (np.array(psig_cs_indices)+num_atoms).tolist()
    
    for r_an, p_an in zip(rsig_cs_atom_map_num, psig_cs_atom_map_num):
        r_idx = GetAtomWithAtomMapNum(rwmol, r_an).GetIdx()
        p_idx = GetAtomWithAtomMapNum(rwmol, p_an).GetIdx()
        for conn in find_connecting_atoms_not_in_sig(input_mol, rsig_match, r_idx):
            rwmol.AddBond(p_idx, conn, input_mol.GetBondBetweenAtoms(r_idx, conn).GetBondType())
            rwmol.RemoveBond(r_idx, conn)
            
    # Remove the atoms from rsig
    for atm_num in rsig_match:
        rwmol.RemoveAtom(GetAtomWithAtomMapNum(rwmol, atm_num).GetIdx())
    
    mol = mol_without_atom_index(Chem.Mol(rwmol)) # with atom number, the molecule ends up invalid after conversion to non-editable(Chem.Mol)
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
        mol = Chem.MolFromSmiles(clean_hydrogen_in_smiles(Chem.MolToSmiles(mol)))

    assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None, "Final mol is not valid"
    assert "." not in Chem.MolToSmiles(mol), "More than 1 molecule in result"
    return mol