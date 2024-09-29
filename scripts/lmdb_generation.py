from fairchem.core.preprocessing import AtomsToGraphs
import ase.io
import lmdb
import pickle
from tqdm import tqdm
import torch

# Define paths for the QM9 dataset
extxyz_file = '../extxyz_files/qm9.extxyz'
csv_file = '../csv_files/qm9.csv'

# Define path to save the LMDB file
lmdb_path = 'qm9_new.lmdb'

# Parse the extxyz file and read the structures
def parse_extxyz_file(extxyz_file):
    try:
        return ase.io.read(extxyz_file, index=':', format='extxyz')
    except Exception as e:
        print(f"Error parsing file {extxyz_file}: {e}")
        return []

# Parse the CSV file and read the energies
def parse_csv_file(csv_file):
    energies = {}
    with open(csv_file, 'r') as file:
        for line in file.readlines()[1:]:  # Skip header line
            parts = line.strip().split(',')
            mol_id = parts[0]
            deltaE = float(parts[-1])  # The last column should be deltaE
            energies[mol_id] = deltaE
    return energies

# Generate LMDB for the QM9 dataset
def generate_lmdb(lmdb_path, extxyz_file, csv_file):
    # Open LMDB environment
    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,
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

    structures = parse_extxyz_file(extxyz_file)
    energies = parse_csv_file(csv_file)

    for system in tqdm(structures, desc=f"Processing QM9 dataset"):
        # Extract the mol_id from the first key in system.info
        mol_id = next(iter(system.info.keys()))
        #print(f"Extracted mol_id: {mol_id}")  # Debugging line
        
        if mol_id is None:
            print(f"mol_id is None, skipping this structure.")
            continue
        
        if mol_id not in energies:
            print(f"mol_id {mol_id} not found in energies, skipping.")
            continue

        data = a2g.convert(system)
        data.sid = int(idx)  # Ensure `sid` is stored as a plain integer
        data.y_relaxed = float(energies[mol_id])  # Ensure `y_relaxed` is stored as a plain float

        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
        db.sync()

        idx += 1

    # Store the length of the dataset
    txn = db.begin(write=True)
    txn.put(b'length', pickle.dumps(idx, protocol=-1))  # Store the total number of entries as the 'length'
    txn.commit()

    db.sync()
    db.close()
    print(f"LMDB creation completed for QM9 at {lmdb_path} with {idx} entries.")

if __name__ == "__main__":
    generate_lmdb(lmdb_path, extxyz_file, csv_file)
