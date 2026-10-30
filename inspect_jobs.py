import os
import pandas as pd

folder = "data/raw"

for file in os.listdir(folder):
    print("\nFile:", file)

    if file.lower().endswith(".csv"):
        df = pd.read_csv(
            os.path.join(folder, file),
            nrows=2
        )

        print("Columns:", df.columns.tolist())