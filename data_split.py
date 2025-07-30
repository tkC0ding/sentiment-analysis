import pandas as pd
import os

data_dir = "data/data.csv"
output_dir = "data/preprocessed_data"

data = pd.read_csv(data_dir)

train_percent = 0.7
val_percent = 0.15
test_percent = 0.15

n = len(data)

train_frac = int(n*train_percent) + 1
test_frac = int(n*test_percent)
val_frac = int(n*val_percent)

train_df = data[:train_frac]
val_df = data[train_frac:train_frac+val_frac]
test_df = data[-test_frac:]

train_df.to_csv(os.path.join(output_dir, "train_df.csv"))
train_df.to_csv(os.path.join(output_dir, "test_df.csv"))
train_df.to_csv(os.path.join(output_dir, "val_df.csv"))