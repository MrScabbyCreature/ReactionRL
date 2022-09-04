'''
This code reads simulator_dataset.csv and extracts all unique mols.
'''
from rdkit import Chem
from rdkit.Chem import AllChem

def get_mol_certificate(mol):
    '''
    Takes a Chem.Mol and returns Morgan fingerprint in base64
    '''
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2).ToBase64()


def mol_with_atom_index(mol):
    '''
    Takes a Chem.Mol without atom indices and return a Chem.Mol with atom indices.
    eg. "CCCCC" -> "[CH3:0][CH2:1][CH2:2][CH2:3][CH3:4]" (in Chem.Mol format)
    '''
    mol = Chem.Mol(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol
