import numpy as np
import random

# Create a 10x10 numpy matrix with random integers between 1 and 5

matrix = np.random.randint(1, 6, size=(3,10,10))
print(matrix)
matrix = matrix[0,:,:]
print(matrix)
# Find the positions (indices) where the matrix has the value 3
positions = np.argwhere(matrix == 3)

# Check if any positions with the value 3 exist
if positions.size > 0:
    # Randomly select one position from the list of satisfied positions
    selected_position = random.choice(positions)

    print(f"Matrix:\n{matrix}")
    print(f"Positions with value 3:\n{positions}")
    print(f"Selected position: {selected_position}")
else:
    print("No positions with the value 3 found.")
