import os
import pandas as pd

# Directory paths
base_dir = './data/train/busi_elastogram'
class_dirs = ['1', '2', '3', '4', '5']

# Adjust the logic to assign labels based on the directory number
data = []

for class_dir in class_dirs:
    full_dir = os.path.join(base_dir, class_dir)
    if os.path.exists(full_dir):
        for filename in os.listdir(full_dir):
            if filename.endswith('.png'):
                label = int(class_dir)  # Label based on the directory number
                data.append([filename, label])

# Create the dataframe
df = pd.DataFrame(data, columns=['name', 'label'])

# Save the dataframe to a CSV file
csv_path = './busi_elastogram_label.csv'
df.to_csv(csv_path, index=False)