# Re-import necessary libraries and re-load the data after state reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'action_record.csv'
data = pd.read_csv(file_path, header=None)

# Define data points
x_coords = data.iloc[:, 0].values
y_coords = data.iloc[:, 1].values

# Generate and display the hexbin scatter density plot
plt.figure(figsize=(10, 8))
plt.hexbin(x_coords, y_coords, gridsize=20, cmap='viridis')

# Add color bar
plt.colorbar(label='Log-scaled Point Density')

# Plot settings
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Hexbin Scatter Density Plot')
plt.show()
