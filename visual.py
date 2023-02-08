import numpy as np
import pandas as pad
import pandas as pd
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn

dataUni = str("Star99999_raw.csv")

#Plx and e_plx are in milliarcsec need to convert into arsec
data = pad.read_csv(dataUni, thousands=",")

#remove Id col
data.drop(columns="Id", inplace=True)

#convert datatype object to float
data["Vmag"] = pd.to_numeric(data["Vmag"], downcast="float", errors="coerce")
data["Plx"] = pd.to_numeric(data["Plx"], downcast="float", errors="coerce")
data["e_Plx"] = pd.to_numeric(data["e_Plx"], downcast="float", errors="coerce")
data["B-V"] = pd.to_numeric(data["B-V"], downcast="float", errors="coerce")

#delete any raw with atleast one NaN
data.dropna(how="any", inplace=True)

data.info()
print(data.isnull().sum())


print(data)
print(data["SpType"].value_counts())

data.drop(data[((data["e_Plx"] * 100) / data["Plx"]) >= 10].index, inplace=True)


data.info()
print("min: {} max: {}".format(data["e_Plx"].min(), data["e_Plx"].max()))

