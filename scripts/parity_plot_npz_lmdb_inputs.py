import numpy as np
import lmdb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to load true values from the LMDB file
def load_true_values(lmdb_path):
    true_values = {}
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key != b'length':  # Skip the 'length' key if it exists
                data = pickle.loads(value)
                true_values[data.sid] = data.y_relaxed
    return true_values

# Paths to the npz file (predictions) and LMDB file (true values)
predictions_npz_path = "../cycle_predictions/cycle_3_predictions.npz"
test_lmdb_path = "../lmdb_files/qm9_curated_80_test.lmdb"

# Load predictions from npz file
predictions_data = np.load(predictions_npz_path)
predicted_energies = predictions_data['energy'].flatten()
predicted_ids = predictions_data['ids']

# Load true values from LMDB
true_values_dict = load_true_values(test_lmdb_path)

# Align predictions and true values by ids
true_energies = []
for sid in predicted_ids:
    true_energies.append(true_values_dict[int(sid)])

true_energies = np.array(true_energies)

# Calculate MAE, MSE, RMSE
mae = mean_absolute_error(true_energies, predicted_energies)
mse = mean_squared_error(true_energies, predicted_energies)
rmse = np.sqrt(mse)

# Parity plot: True vs. Predicted values
plt.figure(figsize=(6, 6))
plt.scatter(true_energies, predicted_energies, alpha=0.5)
plt.plot([min(true_energies), max(true_energies)], [min(true_energies), max(true_energies)], color='red', linestyle='--')

# Add text with MAE, MSE, and RMSE
plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nMSE: {mse:.6f}\nRMSE: {rmse:.4f}',
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Plot formatting
plt.xlabel('True deltaE (Ha)')
plt.ylabel('Predicted deltaE (Ha)')
plt.title('Test Set')
plt.grid(True)
plt.legend()
#plt.savefig('parity_plot_with_metrics.pdf', format='pdf')  # Save the plot as PDF
plt.show()
