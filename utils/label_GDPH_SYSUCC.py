import os
import pandas as pd

# Define the directory containing images and create an empty list to store file data
image_directory = './GDPH&SYSUCC/img'
data = []

# List all files in the directory
for filename in os.listdir(image_directory):
    if "benign" in filename:
        label = 0
    elif "malignant" in filename:
        label = 1
    else:
        continue  # Skip files that do not match the pattern

    data.append({'name': filename, 'label': label})

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(data)
csv_path = './label.csv'
df.to_csv(csv_path, index=False)