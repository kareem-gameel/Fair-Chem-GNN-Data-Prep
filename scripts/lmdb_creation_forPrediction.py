from ase.io import read
from tqdm import tqdm
import lmdb
import pickle
from fairchem.core.preprocessing import AtomsToGraphs

# File paths
extxyz_file = '../extxyz_files/selected_tmqm.extxyz'  # Your extxyz file with mol_id
lmdb_output = '../lmdb_files/your_data_for_prediction.lmdb'  # Output LMDB for predictions

# Read extxyz file for data
raw_data = read(extxyz_file, index=slice(None), format='extxyz')

# Convert Atoms to Data object
a2g = AtomsToGraphs(max_neigh=50, radius=6, r_energy=False, r_forces=False, r_distances=False, r_edges=False, r_fixed=True)

# Convert atoms objects into graphs
data_objects = []
for idx, system in enumerate(raw_data):
    data = a2g.convert(system)
    data.sid = idx
    
    # Extract mol_id from the system.info dictionary (comment line in the extxyz file)
    mol_id = system.info.get('mol_id', f'unknown_{idx}')  # Fallback to 'unknown_{idx}' if mol_id is not present
    data.mol_id = mol_id  # Store mol_id in the Data object

    data_objects.append(data)

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

print("Prediction LMDB generation completed!")
