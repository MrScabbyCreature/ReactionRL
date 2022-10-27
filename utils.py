'''
This code reads simulator_dataset.csv and extracts all unique mols.
'''
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import re
from PIL import Image
from rewards.properties import logP, qed, drd2, similarity
import os

MAIN_DIR = os.getenv('MAIN_DIR')

def get_mol_certificate(mol):
    '''
    Takes a Chem.Mol and returns Morgan fingerprint in base64
    '''
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBase64()

def clean_hydrogen_in_smiles(smiles):
    '''
    Some clean-ups Idk how to do in molecule. So I do it in smiles after conversion.
    1. Remove extra hydrogens for even sized rings
    '''
    return re.sub("\[([a-zA-Z])H[0-9]\]", r"\1", smiles)


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

def mol_without_atom_index(mol):
    '''
    Convert smiles with numbers to smiles without numbers
    '''
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).ClearProp( 'molAtomMapNumber' )
    return mol


def find_connecting_atoms_not_in_sig(mol, sig_indices, centre):
    cen_atom = mol.GetAtomWithIdx(centre)
    neighbors_indices = list(map(lambda x: x.GetIdx(), cen_atom.GetNeighbors()))
    return set(neighbors_indices) - set(sig_indices)

def GetAtomWithAtomMapNum(mol, num):
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == num:
            return atom
    return None

def calc_reward(state, action, next_state, metric='logp'):
    '''
    Get the reward based on some Chemical metric (logp, QED, DRD2)
    '''
    if metric == "logp":
        return logP(next_state) - logP(state)
    elif metric == "qed":
        return qed(next_state) - qed(state)
    elif metric == "drd2":
        return drd2(next_state) - drd2(state)
    else:
        raise f"Reward metric {metric} not found."
    

def display_mol(mol, title=None):
    im = Draw.MolToImage(mol)
    im.show(title)

def get_concat_v_multi_resize(im_list, resample=Image.Resampling.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=resample)
                    for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst