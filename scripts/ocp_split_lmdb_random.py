import sys
from fairchem.core.datasets import LmdbDataset
import lmdb
import pickle
from tqdm import trange
import numpy as np

# File paths
dataset_path = "../lmdb_files/qm9_curated.lmdb"
train_lmdb_path = "../lmdb_files/qm9_curated_80_train.lmdb"
val_lmdb_path = "../lmdb_files/qm9_curated_80_val.lmdb"
test_lmdb_path = "../lmdb_files/qm9_curated_80_test.lmdb"

# Get a random number generator
rng = np.random.default_rng()

# Read the dataset you want to split
dataset = LmdbDataset({"src": dataset_path})

# Want val_frac and test_frac to be relatively small to prevent long queue times due to system trying to guess a new random integer
val_frac = 0.1
test_frac = 0.1

val_cutoff = round(val_frac * len(dataset))
test_cutoff = round(test_frac * len(dataset))

used_indexes = []  # Used to track which structures are already in a dataset

# Writing val dataset
db2 = lmdb.open(
    val_lmdb_path,
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

print("Creating val dataset")

val = []
i = 0
while i < val_cutoff:
    id = rng.integers(0, len(dataset))
    if id not in used_indexes:
        used_indexes.append(id)
        val.append(dataset[id])
        i += 1

for id in trange(len(val)):
    data = val[id]
    data.val_id = id
    txn = db2.begin(write=True)
    txn.put(f"{id}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()
    db2.sync()

db2.close()

# Writing test dataset
db3 = lmdb.open(
    test_lmdb_path,
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

print("Creating test dataset")

test = []
j = 0
while j < test_cutoff:
    id = rng.integers(0, len(dataset))
    if id not in used_indexes:
        used_indexes.append(id)
        test.append(dataset[id])
        j += 1

for id in trange(len(test)):
    data = test[id]
    data.test_id = id
    txn = db3.begin(write=True)
    txn.put(f"{id}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()
    db3.sync()

db3.close()

# Writing train dataset
db = lmdb.open(
    train_lmdb_path,
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

print("Creating train dataset")

train_id = 0
for id in trange(len(dataset)):
    if id not in used_indexes:
        data = dataset[id]
        data.train_id = train_id
        txn = db.begin(write=True)
        txn.put(f"{train_id}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
        db.sync()

        train_id += 1

db.close()

print("Done!")
