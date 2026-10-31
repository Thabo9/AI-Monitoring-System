import pandas as pd
import numpy as np


def clean_data(df):

 df = df.copy()

# 1. Fix data types from float to integer
 df["faulty"] = df["faulty"].astype(int)
 df[["equipment", "location"]] = df[["equipment", "location"]].astype("category")

# 2. Fixing negative vibration values
 bad_vib = df["vibration"] < 0
 print(f"Negative vibration rows: {bad_vib.sum()}")
 df.loc[bad_vib, "vibration"] = np.nan
 df["vibration"] = df["vibration"].fillna(
df.groupby("equipment", observed=True)["vibration"].transform("median"))


 assert df.isna().sum().sum() == 0
 assert (df["vibration"] >= 0).all()
 assert df["faulty"].isin([0, 1]).all()
 df.to_csv("equipment_clean.csv")


 return df