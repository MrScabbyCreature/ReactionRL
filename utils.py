'''
This code reads simulator_dataset.csv and extracts all unique mols.
'''
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import re
from PIL import Image, ImageDraw
from rewards.properties import logP, qed, drd2, similarity, SA
import os
import numpy as np

MAIN_DIR = os.getenv('MAIN_DIR')

class bcolors: # For printing colored text
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_mol_certificate(mol):
    '''
    Takes a Chem.Mol and returns Morgan fingerprint in base64
    '''
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBase64()

def clean_hydrogen_in_smiles(smiles):
    '''
    Some clean-ups Idk how to do in molecule. So I do it in smiles after conversion.
    1. Remove extra H - But odd sized aromatic rings need 1 atom with explicitly specified H's
    '''
    s1 = re.sub("\[([^\]])H[0-9]?\]", r"\1", smiles) 
    if Chem.MolFromSmiles(s1) is None: # Got rid of H from odd aromatic ring
        s2 = re.sub("\[([A-Z])H[0-9]?\]", r"\1", smiles)
        return s2
    return s1


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

def calc_reward(state, action, next_state, target=None, metric='logp'):
    '''
    Get the reward based on some Chemical metric (logp, QED, DRD2)
    '''
    if metric == "logp":
        return logP(next_state) - logP(state)
    elif metric == "qed":
        return qed(next_state) - qed(state)
    elif metric == "drd2":
        return drd2(next_state) - drd2(state)
    elif metric == "SA":
        def _transform(sascore):
            '''
            SA is between 1 and 10 where 1 is easily synthesizable.
            log(11-SA) is between 0 and 1 where 1 is easily synthesizable (so fits as a reward metric).
            Also, log(11-SA) has a steeper curve at 10 than at 1, so maximising the reward at 10 should be "more important" - According to some research SA between 1 and 4 is good, so don't care much when it reaches that point.
            '''
            return np.log10(11-sascore)
        return _transform(SA(next_state)) - _transform(SA(state))
    elif metric == "sim":
        assert target is not None, "Need a target for similirity reward"
        return similarity(next_state, target) - similarity(state, target)
    else:
        raise f"Reward metric {metric} not found."
    

def display_mol(mol, title=None):
    im = Draw.MolToImage(mol)
    im.show(title)

def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
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

def get_concat_h_blank(im1, im2, color=(255, 255, 255)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, im2.height//2 - im1.height//2))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(255, 255, 255)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (im2.width//2 - im1.width//2, 0))
    dst.paste(im2, (0, im1.height))
    return dst