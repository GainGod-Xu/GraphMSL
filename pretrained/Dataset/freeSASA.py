from rdkit import Chem
from rdkit.Chem import rdFreeSASA
import torch
import pickle
import os

def calc_freeSASA(mol):
    probe_radius_list = [round(0.05 + i * 0.05, 2) for i in range(128)]
    results = []

    # Classify atoms
    radii = rdFreeSASA.classifyAtoms(mol)

    for pr in probe_radius_list:
        # Set up SASA options
        sasa_opts = rdFreeSASA.SASAOpts()
        sasa_opts.probeRadius = pr  # Set probe radius
        sasa_opts.algorithm = rdFreeSASA.ShrakeRupley

        # Calculate SASA
        sasa = rdFreeSASA.CalcSASA(mol, radii=radii, opts=sasa_opts)
        results.append(sasa)
    return results

def read(sdf):
    folder_name = "surface"
    # Check if the folder exists, if not create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    mol_supplier = Chem.SDMolSupplier(sdf, removeHs=False)
    for idx, mol in enumerate(mol_supplier, start=0):
        if mol:
            try:
                props = mol.GetPropsAsDict()
                #print("Properties of molecule", idx)
                cid = "{:08d}".format(props['PUBCHEM_COMPOUND_CID'])
                conf_id = props['PUBCHEM_CONFORMER_ID']
                id = str(cid) + "_" + str(conf_id)
                sasa_results = calc_freeSASA(mol)
                sasa_tensor = torch.tensor(sasa_results)
                file_path = os.path.join(folder_name, f'{id}.pickle')
                with open(file_path, 'wb') as f:
                    pickle.dump(sasa_tensor, f)
            except Exception as e:
                print(f"Error processing molecule {idx}: {e}")


def main():
    sdfs = os.listdir('sdf')
    for sdf in sdfs:
        read(sdf)

if __name__ == "__main__":
    main()