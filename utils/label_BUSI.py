import os
import pandas as pd

def create_label_csv(data_dir, output_csv_path):
    labels = []
    filenames = sorted(os.listdir(data_dir))

    for filename in filenames:
        if filename.startswith('normal'):
            label = 0
        elif filename.startswith('benign'):
            label = 1
        elif filename.startswith('malignant'):
            label = 2
        else:
            continue
        labels.append({'name': filename, 'label': label})

    df = pd.DataFrame(labels)
    df = df.sort_values(by='label')
    df.to_csv(output_csv_path, index=False)

data_dir = './data/BUSI/test/img'
output_csv_path = './data/BUSI/test/label.csv'
create_label_csv(data_dir, output_csv_path)