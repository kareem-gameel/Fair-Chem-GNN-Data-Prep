import lmdb
import pickle
from io import BytesIO
import sys

# Check if the file path is provided as an argument
if len(sys.argv) < 2:
    print("Usage: python inspect_lmdb.py <lmdb_file_path>")
    sys.exit(1)

# Path to your LMDB file from the command line argument
lmdb_path = sys.argv[1]

# Open the LMDB environment
env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)

# Start a read transaction
with env.begin() as txn:
    # Initialize a counter
    num_entries = 0

    # Create a cursor to iterate over the items in the database
    cursor = txn.cursor()
    for key, value in cursor:
        if key != b'length':  # Skip the 'length' key if it exists
            num_entries += 1

    # Print the total number of entries found
    print(f"Total number of entries: {num_entries}")

    # Optionally, check and print the first entry to see the data structure
    first_entry = txn.get(b'0')
    if first_entry is not None:
        data = pickle.loads(first_entry)
        print("First entry found in the LMDB:", data)
    else:
        print("No entries found in the LMDB.")
