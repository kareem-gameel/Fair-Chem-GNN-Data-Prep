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

from fairchem.core.preprocessing import AtomsToGraphs  # Importing the AtomsToGraphs class to convert structures into graph representations.
import ase.io  # Importing ASE (Atomic Simulation Environment) to handle extxyz file parsing.
import lmdb  # Importing the lmdb library for creating a fast key-value database (LMDB).
import pickle  # Importing pickle to serialize data for storage in the LMDB.
from tqdm import tqdm  # Importing tqdm to show a progress bar for the generation process.
import torch  # Importing torch to handle tensor operations for attributes such as `sid` and `y_relaxed`.

# Define paths for the QM9 dataset
extxyz_file = '../extxyz_files/qm9.extxyz'  # Path to the extxyz file containing molecular structures.
csv_file = '../csv_files/qm9_deltaE.csv'  # Path to the CSV file containing mol_id and deltaE values.
lmdb_path = 'qm9_deltaE.lmdb'  # Path to save the LMDB file output.

# Function to parse the extxyz file and read the molecular structures.
def parse_extxyz_file(extxyz_file):
    try:
        # ase.io.read reads all structures in the extxyz file. The ':' index indicates we want to read all entries.
        return ase.io.read(extxyz_file, index=':', format='extxyz')
    except Exception as e:
        # If there's an error reading the file, we catch the exception and print the error message.
        print(f"Error parsing file {extxyz_file}: {e}")
        return []  # Return an empty list if there's an error.

# Function to parse the CSV file and extract mol_id and deltaE values.
def parse_csv_file(csv_file):
    energies = {}  # Initialize an empty dictionary to store mol_id and deltaE values.
    with open(csv_file, 'r') as file:
        # Read all lines in the CSV file starting from the second line (skip header).
        for line in file.readlines()[1:]:  
            parts = line.strip().split(',')  # Split each line by commas to get mol_id and deltaE.
            mol_id = parts[0]  # The first column is mol_id.
            deltaE = float(parts[-1])  # The last column is deltaE, converted to a float.
            energies[mol_id] = deltaE  # Store the mol_id and deltaE in the dictionary.
    return energies  # Return the dictionary containing mol_id and deltaE.

# Function to generate the LMDB for the QM9 dataset.
def generate_lmdb(lmdb_path, extxyz_file, csv_file):
    # Open an LMDB environment for writing data. This initializes the LMDB file where data will be stored.
    db = lmdb.open(
        lmdb_path,  # Path to the LMDB file.
        map_size=1099511627776 * 2,  # Maximum size of the LMDB (2 terabytes in this case).
        subdir=False,  # Whether the path should be treated as a directory or a regular file.
        meminit=False,  # Don't initialize memory when opening the LMDB.
        map_async=True,  # Allow asynchronous flushing to the LMDB file.
    )

    idx = 0  # Initialize an index counter for assigning unique system identifiers (sid).

    # Initialize the AtomsToGraphs object for converting ASE atoms objects into graph data.
    a2g = AtomsToGraphs(
        max_neigh=50,  # Maximum number of neighbors for graph representation.
        radius=6,  # Radius cutoff for determining neighbors.
        r_energy=False,  # Energy information is not read from the structure.
        r_forces=False,  # Forces are not included.
        r_distances=False,  # Distances between atoms are not read.
        r_edges=False,  # Edge data (bond information) is not used.
        r_fixed=False,  # Atom constraints are not considered.
    )

    # Parse the extxyz file to get the list of molecular structures.
    structures = parse_extxyz_file(extxyz_file)
    # Parse the CSV file to get the mol_id and deltaE values.
    energies = parse_csv_file(csv_file)

    # Loop through each mol_id and deltaE value from the CSV file.
    for mol_id, deltaE in tqdm(energies.items(), desc="Processing QM9 dataset"):
        # Try to match the mol_id with the corresponding structure from the extxyz file.
        try:
            # Fetch the structure using the index (idx), assuming the structures and CSV file are aligned.
            system = structures[idx]

            # Convert the ASE atoms object (system) into a graph representation using AtomsToGraphs.
            data = a2g.convert(system)
            # Assign a unique identifier (sid) to the graph data.
            data.sid = torch.LongTensor([idx])
            # Assign the deltaE value as the target (y_relaxed) for this graph.
            data.y_relaxed = torch.tensor([deltaE])

            # Commented out the debug message to avoid printing during the process.
            # print(f"Processing mol_id {mol_id} with deltaE {deltaE}")

            # Begin an LMDB transaction to write data.
            txn = db.begin(write=True)
            # Store the serialized (pickled) graph data in the LMDB, using the index (idx) as the key.
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()  # Commit the transaction to save the data.
            db.sync()  # Ensure that all data is flushed to disk.

            # Increment the index counter for the next structure.
            idx += 1

        # Catch any errors that occur during processing and print an error message.
        except Exception as e:
            print(f"Error processing system with mol_id {mol_id}: {e}")
            continue  # Skip to the next iteration if an error occurs.

    # Close the LMDB database after all data has been processed and saved.
    db.close()
    # Print a message indicating that LMDB generation has completed.
    print(f"LMDB creation completed for QM9 at {lmdb_path}")

# Main entry point: Calls the generate_lmdb function if the script is executed directly.
if __name__ == "__main__":
    # Call the LMDB generation function with the defined paths for LMDB, extxyz, and CSV files.
    generate_lmdb(lmdb_path, extxyz_file, csv_file)
