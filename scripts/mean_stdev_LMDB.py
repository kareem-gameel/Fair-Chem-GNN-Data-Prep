import lmdb
import pickle
import numpy as np
import sys

# Check if the file path is provided as an argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <lmdb_file_path>")
    sys.exit(1)

# Define the path to the LMDB file from the command line argument
lmdb_path = sys.argv[1]

# Open the LMDB environment
env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)

# Initialize a list to hold energy values
energy_values = []

# Start a read transaction
with env.begin() as txn:
    # Get the length of the dataset (number of entries)
    length = pickle.loads(txn.get(b'length'))

    # Iterate over the LMDB entries
    for idx in range(length):
        key = f"{idx}".encode("ascii")  # Encode the index as a key
        entry = txn.get(key)
        if entry is not None:
            data = pickle.loads(entry)  # Load the data object
            energy = data.y_relaxed  # Extract the relaxed energy value
            energy_values.append(energy)  # Append the energy to the list

# Convert the list to a numpy array
energy_array = np.array(energy_values)

# Calculate mean and standard deviation
mean_energy = np.mean(energy_array)
std_energy = np.std(energy_array)

print(f"Mean energy: {mean_energy}")
print(f"Standard deviation: {std_energy}")
