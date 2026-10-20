
import pandas as pd

file_path = "data/dataset.xlsx"

df = pd.read_excel(file_path)

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes)
