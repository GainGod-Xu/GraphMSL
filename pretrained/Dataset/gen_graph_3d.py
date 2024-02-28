import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np
import os
import pandas as pd
import pickle
import argparse

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)), ### changed
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_atom_partial_charge': (-1.00000, 1.00000),
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE
         
    ]
}


def mol_to_graph_data_obj_simple(molecule, max_attempts=5):

    if molecule is None:
        raise ValueError("Invalid SMILES string")

    # Get atom features (atomic number, chirality, hybridization)
    atom_features_list = []
    for atom in molecule.GetAtoms():
        atom_feature = \
        [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
        [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
        [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())]

        atom_features_list.append(atom_feature)
        #print(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    
    conformer=mol.GetConformer()
    # Get atom positions (coordinates)
    pos = torch.tensor([list(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())], dtype=torch.float)
    #print(pos)
    # Get edge indices
    num_bond_features = 2
    if len(molecule.GetBonds()) > 0:
        edge_list = []
        edge_feature_list = []
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(
                    bond.GetBondDir())]
            edge_list.append((i, j))
            edge_feature_list.append(edge_feature)
            edge_list.append((j, i))  # add both directions for undirected graph
            edge_feature_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feature_list), dtype=torch.long)

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return data



def save_molecules_as_graph_data(sdf_file):
    folder_path = "graph_3d"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mol_supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)

    for idx, mol in enumerate(mol_supplier, start=0):
        try:
            # Convert the molecule to a Data object
            if mol:
                props = mol.GetPropsAsDict()
                cid = "{:08d}".format(props['PUBCHEM_COMPOUND_CID'])
                conf_id = props['PUBCHEM_CONFORMER_ID']
                id = str(cid) + "_" + str(conf_id)
                file_path = os.path.join(folder_path, f'{id}.pickle')
                data = mol_to_graph_data_obj_simple(mol)  # Define this function

                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)

        except AttributeError as e:
            print(f"AttributeError processing molecule {id}: {e}")
            # Print molecule details if needed, e.g., mol.GetPropsAsDict()
        except Exception as e:
            print(f"Error processing molecule {id}: {type(e).__name__}: {e}")


def main():
    sdfs = os.listdir('sdf')
    for sdf in sdfs:
        save_molecules_as_graph_data(sdf)


if __name__ == "__main__":
    main()


  
    