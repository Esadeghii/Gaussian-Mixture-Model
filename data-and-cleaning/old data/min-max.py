import pandas as pd
import numpy as np

def min_max_normalize(data, min_val=0, max_val=31.2934377667882):
    return (data - min_val) / (max_val - min_val)

# Load the Excel file
file_path = 'test.xlsx'  # Replace with your file path
df = pd.read_excel(file_path, header=None)  # Assuming no header in the file

# Apply min-max normalization with specified min and max values
normalized_df = df.applymap(lambda x: min_max_normalize(x))

# Save to a new Excel file
normalized_file_path = 'normalized_file.xlsx'  # Replace with desired new file path
normalized_df.to_excel(normalized_file_path, index=False, header=None)
