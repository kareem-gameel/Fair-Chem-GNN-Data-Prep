import lmdb
import pickle
from tqdm import trange
import numpy as np
from fairchem.core.datasets import LmdbDataset

# Set the file paths
original_lmdb = "../lmdb_files/qm9_subset.lmdb"
train_lmdb = "../lmdb_files/qm9_subset_80_train.lmdb"
val_lmdb = "../lmdb_files/qm9_subset_80_val.lmdb"
test_lmdb = "../lmdb_files/qm9_subset_80_test.lmdb"

# Set random number generator
rng = np.random.default_rng()

# Load the dataset using FairChem LmdbDataset
dataset = LmdbDataset({"src": original_lmdb})

# Set the split fractions (adjust if necessary)
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

# Calculate the cutoffs for each split
train_cutoff = round(train_frac * len(dataset))
val_cutoff = round(val_frac * len(dataset))
test_cutoff = round(test_frac * len(dataset))

# Track used indexes to avoid overlap
used_indexes = []

# Function to create a new LMDB file
def create_lmdb(new_lmdb_path, data_list, label):
    db = lmdb.open(new_lmdb_path, map_size=1099511627776 * 2, subdir=False, meminit=False, map_async=True)
    print(f"Creating {label} dataset")
    for idx in trange(len(data_list)):
        data = data_list[idx]
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
        db.sync()
    db.close()

# Generate validation set
val = []
i = 0
while i < val_cutoff:
    id = rng.integers(0, len(dataset))
    if id not in used_indexes:
        used_indexes.append(id)
        val.append(dataset[id])
        i += 1

create_lmdb(val_lmdb, val, "val")

# Generate test set
test = []
j = 0
while j < test_cutoff:
    id = rng.integers(0, len(dataset))
    if id not in used_indexes:
        used_indexes.append(id)
        test.append(dataset[id])
        j += 1

create_lmdb(test_lmdb, test, "test")

# Generate train set
train = []
for id in trange(len(dataset)): 
    if id not in used_indexes:
        train.append(dataset[id])

create_lmdb(train_lmdb, train, "train")

print("Splitting complete!")
