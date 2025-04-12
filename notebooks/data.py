import pandas as pd
import os

DATA_PATH = "../data/bottle.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found!")

# Load datasets
cast_df = pd.read_csv("../data/cast.csv", low_memory=False,encoding="ISO-8859-1")
bottle_df = pd.read_csv("../data/bottle.csv", low_memory=False,encoding="ISO-8859-1")

# Merge datasets on 'Cst_Cnt'
df = bottle_df.merge(cast_df, on='Cst_Cnt', suffixes=('_bottle', '_cast'))
print("Dataset loaded successfully!")

# Sezione 3: Info generale
print("\n Dimensions:", df.shape)
print("\n Columns:")
print(df.columns.tolist())

df.info(show_counts=True,verbose=True)