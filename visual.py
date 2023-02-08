import numpy as np
import pandas as pad
import pandas as pd
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn

dataUni = str("Star99999_raw.csv")
data = pad.read_csv(dataUni, thousands=",")
data.drop(columns="Id", inplace=True)
data["Vmag"] = pd.to_numeric(data["Vmag"], downcast="float", errors="coerce")
data["Plx"] = pd.to_numeric(data["Plx"], downcast="float", errors="coerce")
data["e_Plx"] = pd.to_numeric(data["e_Plx"], downcast="float", errors="coerce")
data["B-V"] = pd.to_numeric(data["B-V"], downcast="float", errors="coerce")

data.dropna(how="any", inplace=True)

data.info()
print(data.isnull().sum())

print(data)