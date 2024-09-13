import numpy as np

# Original 3D array of shape (3, 3, 3) filled with zeros
original_array = np.ones((4, 4, 4), dtype=int)

# New 2D array of shape (3, 3) to replace the first slice
new_slice = np.ones((4, 4), dtype=int)

# Replicate the new slice along the third dimension
new_slice_expanded = np.stack([new_slice] * 4, axis=2)

# Replace the first slice in the original array
original_array[2] = new_slice_expanded[:, :, 0] + original_array[2]

print(original_array)
