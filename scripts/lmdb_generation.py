"""
This script generates an LMDB file for the QM9 dataset using the FairChem library.

- The script reads molecular structures from an extxyz file and corresponding deltaE values from a CSV file.
- It uses the AtomsToGraphs class from the FairChem library to convert each molecular structure into a graph representation.
- The mol_id from the CSV file is used to identify each molecule, and the corresponding deltaE is stored as y_relaxed.
- The output is an LMDB file containing graph representations of the molecular structures and their associated energies.

Steps:
1. Parse the extxyz file to retrieve molecular structures.
2. Parse the CSV file to retrieve mol_id and deltaE values.
3. Match each mol_id with its structure, convert the structure to a graph, and store the result in an LMDB.
"""

from fairchem.core.preprocessing import AtomsToGraphs
import ase.io
import lmdb
import pickle
from tqdm import tqdm
import torch

# Define paths for the QM9 dataset
extxyz_file = '../extxyz_files/qm9.extxyz'
csv_file = '../csv_files/qm9_deltaE.csv'
lmdb_path = 'qm9_deltaE.lmdb'  # Path to save the LMDB file

# Parse the extxyz file and read the structures
def parse_extxyz_file(extxyz_file):
    try:
        return ase.io.read(extxyz_file, index=':', format='extxyz')
    except Exception as e:
        print(f"Error parsing file {extxyz_file}: {e}")
        return []

# Parse the CSV file and read the energies and mol_ids
def parse_csv_file(csv_file):
    energies = {}
    with open(csv_file, 'r') as file:
        for line in file.readlines()[1:]:  # Skip header line
            parts = line.strip().split(',')
            mol_id = parts[0]
            deltaE = float(parts[-1])  # Assuming deltaE is the last column
            energies[mol_id] = deltaE
    return energies

# Generate LMDB for the QM9 dataset
def generate_lmdb(lmdb_path, extxyz_file, csv_file):
    # Open LMDB environment
    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,  # Adjust size if needed
        subdir=False,
        meminit=False,
        map_async=True,
    )

    idx = 0
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=False,
        r_fixed=False,
    )

    # Read structures and energies
    structures = parse_extxyz_file(extxyz_file)
    energies = parse_csv_file(csv_file)

    # Use the mol_id from the CSV to match with the extxyz file
    for mol_id, deltaE in tqdm(energies.items(), desc="Processing QM9 dataset"):
        # Try to match mol_id with the structures list
        try:
            # Match by index assuming structures and CSV are aligned
            system = structures[idx]

            # Convert the system into a graph representation
            data = a2g.convert(system)
            data.sid = torch.LongTensor([idx])
            data.y_relaxed = torch.tensor([deltaE])  # Store deltaE as y_relaxed

            # Commenting out the debug message, keeping only the progress bar
            # print(f"Processing mol_id {mol_id} with deltaE {deltaE}")

            # Write to LMDB
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
            db.sync()

            idx += 1

        except Exception as e:
            print(f"Error processing system with mol_id {mol_id}: {e}")
            continue

    db.close()
    print(f"LMDB creation completed for QM9 at {lmdb_path}")

if __name__ == "__main__":
    generate_lmdb(lmdb_path, extxyz_file, csv_file)
