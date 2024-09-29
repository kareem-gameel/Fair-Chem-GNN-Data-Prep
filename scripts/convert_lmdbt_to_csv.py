import lmdb
import pickle
import csv
from tqdm import tqdm

# Define the path to the LMDB file
lmdb_path = "qm9_new.lmdb"
# Define the output CSV file path
output_csv_path = "qm9_6_fromLMDB.csv"

# Open the LMDB environment
env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)

# Prepare to read from the LMDB
with env.begin() as txn:
    # Get the total number of entries in the LMDB
    length = txn.get(b'length')
    if length is not None:
        length = pickle.loads(length)
        print(f"Total entries: {length}")

    # Open a new CSV file for writing
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['mol_id', 'deltaE'])

        # Iterate over each item in the LMDB
        for idx in tqdm(range(length), desc="Reading LMDB"):
            raw_data = txn.get(f"{idx}".encode())
            if raw_data is not None:
                data = pickle.loads(raw_data)
                # Extract mol_id and deltaE
                mol_id = data.mol_id if hasattr(data, 'mol_id') else 'NA'
                deltaE = data.y_relaxed if hasattr(data, 'y_relaxed') else 'NA'
                # Write to CSV
                writer.writerow([mol_id, deltaE])
            else:
                print(f"Entry {idx} is missing.")

print("Conversion completed. Data written to:", output_csv_path)
