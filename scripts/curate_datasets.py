import pandas as pd
from ase.io import read, write

# Input and output file paths
extxyz_file = '../../extxyz_files/tmqm_neutral_updated.extxyz'
csv_file = '../../csv_files/tmqm_deltaE.csv'
output_extxyz_file = '../../extxyz_files/tmqm_curated.extxyz'
output_csv_file = '../../csv_files/tmqm_curated_deltaE.csv'

# Step 1: Load the CSV file and extract mol_ids
csv_data = pd.read_csv(csv_file)
csv_mol_ids = set(csv_data['mol_id'])  # Set of mol_ids in the CSV

# Step 2: Load the extxyz file and extract mol_ids
raw_data = read(extxyz_file, index=slice(None), format='extxyz')
extxyz_mol_ids = [system.info['mol_id'] for system in raw_data]  # List of mol_ids in extxyz

# Step 3: Find common `mol_id`s between CSV and extxyz
common_mol_ids = set(extxyz_mol_ids).intersection(csv_mol_ids)  # Common mol_ids in both files

# Step 4: Filter the CSV file to keep only the common mol_ids
filtered_csv_data = csv_data[csv_data['mol_id'].isin(common_mol_ids)]  # Filter CSV based on common mol_ids

# Reorder the CSV to match the order of `mol_id`s in the extxyz file
filtered_extxyz_mol_ids = [mol_id for mol_id in extxyz_mol_ids if mol_id in common_mol_ids]
filtered_csv_data.set_index('mol_id', inplace=True)
filtered_csv_data = filtered_csv_data.loc[filtered_extxyz_mol_ids]  # Reorder CSV based on extxyz mol_ids order

# Step 5: Filter the extxyz data based on common mol_ids (already aligned)
filtered_extxyz_data = [system for system in raw_data if system.info['mol_id'] in common_mol_ids]

# Step 6: Save the filtered and ordered extxyz data
write(output_extxyz_file, filtered_extxyz_data, format='extxyz')

# Step 7: Save the filtered and reordered CSV data
filtered_csv_data.to_csv(output_csv_file)

# Step 8: Final check to ensure both files have the same number of molecules and order
if len(filtered_extxyz_data) == len(filtered_csv_data):
    print(f"Number of molecules in both files: {len(filtered_extxyz_data)}")
    
    # Check if the order of mol_ids matches between the two files
    extxyz_mol_ids_final = [system.info['mol_id'] for system in filtered_extxyz_data]
    csv_mol_ids_final = filtered_csv_data.index.tolist()

    if extxyz_mol_ids_final == csv_mol_ids_final:
        print("The mol_id order matches exactly between the extxyz and CSV files.")
    else:
        print("Error: The mol_id order does not match between the extxyz and CSV files.")
else:
    print(f"Error: Number of molecules in extxyz file ({len(filtered_extxyz_data)}) does not match number of molecules in CSV file ({len(filtered_csv_data)}).")
