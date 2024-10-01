#WARNING: this code assumes matching order and number of mol_ids across the graph features (in the input .extxyz file) 
#and the target energies (in the input .csv file)

from ase.io import read
from tqdm import tqdm
import csv
import lmdb
import pickle
import sys
from fairchem.core.preprocessing import AtomsToGraphs

# File paths
extxyz_file = '../extxyz_files/tmqm_curated.extxyz'
csv_file = '../csv_files/tmqm_deltaE_curated.csv'
lmdb_output = '../lmdb_files/tmqm_curated.lmdb'

# Read extxyz file for data
raw_data = read(extxyz_file, index=slice(None), format='extxyz')

# Convert Atoms to Data object
a2g = AtomsToGraphs(max_neigh=50, radius=6, r_energy=False, r_forces=False, r_distances=False, r_edges=False, r_fixed=True)

# Convert atoms objects into graphs
data_objects = []
for idx, system in enumerate(raw_data):
    data = a2g.convert(system)
    data.sid = idx
    data_objects.append(data)

# Read CSV file of targets, append onto data structure
with open(csv_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            data_objects[line_count-1].mol_id = row[0]
            data_objects[line_count-1].formula = row[1]
            data_objects[line_count-1].deltaE = float(row[2])
            data_objects[line_count-1].y_relaxed = float(row[2])  # Need target to be named y_relaxed
            line_count += 1

# Write LMDB
db = lmdb.open(
    lmdb_output,
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

idx = 0
for sid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    # Assign sid
    data.sid = idx

    txn = db.begin(write=True)
    txn.put(f"{sid}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()
    db.sync()

    idx += 1

db.close()

print("Done!")
