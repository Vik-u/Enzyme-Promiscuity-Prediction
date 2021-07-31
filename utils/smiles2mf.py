from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import numpy as np

def convert(smiles):
    # get the molecule from smiles
    mol = Chem.MolFromSmiles(smiles)
    # get the molecular fingerprint as a 2048 length binary vector from molecule
    mf = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    # convert the vector to a numpy array
    array = np.zeros(1) # the destination array
    ConvertToNumpyArray(mf, array)
    return array


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", help="convert the smiles to morgan fingerprint of radius 2")
    args = parser.parse_args()
    
    mfr2 = convert(args.smiles) 
    print(len(mfr2))
    print(np.unique(mfr2, return_counts=True))
    
