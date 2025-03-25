import numpy as np
import pandas as pad
import pandas as pd
import sklearn
import seaborn as sns
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


dataUni = str("Star99999_raw.csv")

# Plx and e_plx are in milliarcsec need to convert into arsec
data = pad.read_csv(dataUni, thousands=",")

# remove Id col
data.drop(columns="Id", inplace=True)

# convert datatype object to float
data["Vmag"] = pd.to_numeric(data["Vmag"], downcast="float", errors="coerce")
data["Plx"] = pd.to_numeric(data["Plx"], downcast="float", errors="coerce")
data["e_Plx"] = pd.to_numeric(data["e_Plx"], downcast="float", errors="coerce")
data["B-V"] = pd.to_numeric(data["B-V"], downcast="float", errors="coerce")


# delete any raw with atleast one NaN
data.dropna(how="any", inplace=True)

data.info()
print(data.isnull().sum())

print(data)
print(data["SpType"].value_counts())


def func1(x):
    return x / 1000


def sptClass(x: str):
    if x.__contains__('M'):
        return 6
    elif x.__contains__('K'):
        return 5
    elif x.__contains__('G'):
        return 4
    elif x.__contains__('F'):
        return 3
    elif x.__contains__('A'):
        return 2
    elif x.__contains__('B'):
        return 1
    elif x.__contains__('O'):
        return 0


def target(x):
    if x.__contains__('V'):
        return 0
    elif x.__contains__('I'):
        return 1
    return np.nan


'''data['Plx'] = data['Plx'].map(lambda x: func1(x))
data['e_Plx'] = data['e_Plx'].map(lambda x: func1(x))'''

data.drop(data[((data["e_Plx"] * 100) / data["Plx"]) >= 11].index, inplace=True)

Amag = []

for index, row in data.iterrows():
    result = row['Vmag'] + 5 * (math.log10(1 / (math.fabs(row['Plx']/1000))) + 1)

    Amag.append(round(result, 2))


data['Amag'] = Amag
del Amag

data['TargetClass'] = data['SpType'].map(lambda x: target(x))
data['SpType'] = data['SpType'].map(lambda x: sptClass(x))

data.dropna(how="any", inplace=True)


data.info()
print("min: {} max: {}".format(data["e_Plx"].min(), data["e_Plx"].max()))

smote = SMOTE(random_state=42)

X, Y = data.iloc[:, :5], data.iloc[:, 6]
stdc = StandardScaler()
X_std = stdc.fit_transform(X)




#X_train_smote, X_test_smote, Y_train_smote, Y_test_smote = train_test_split(X_smote, Y_smote, test_size=0.3, random_state=42)


X_smote, Y_smote = smote.fit_resample(X_std, Y)
#X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, test_size=0.3, random_state=42)


'''
stdc = StandardScaler()
X_train_std = stdc.fit_transform(X_smote)
X_test = stdc.transform(X_test)
'''


print('Original dataset shape %s' % Counter(Y))
print('Resampled dataset shape %s' % Counter(Y_smote))


'''
tree_model = DecisionTreeClassifier(max_depth=40)
tree_model.fit(X_train_std, Y_smote)
Y_pred = tree_model.predict(X_test)
labels = np.unique(Y_test)

print(confusion_matrix(Y_test, Y_pred, labels=labels))
print(classification_report(Y_test, Y_pred))

print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred)
'''


#sns.boxplot( data=data)
#plt.show()
data.plot(kind='box', subplots=True, layout=(2,4), figsize=(15,18))
plt.show()


kfold = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
tree_model = RandomForestClassifier(max_depth=30)
scores = []

for k, (train_index, test_index) in enumerate(kfold.split(X_smote, Y_smote)):
    tree_model.fit(X_smote[train_index], Y_smote[train_index])
    Y_pred = tree_model.predict(X_smote[test_index])
    labels = np.unique(Y_smote[test_index])
    conMat = confusion_matrix(Y_smote[test_index], Y_pred, labels=labels)
    report = classification_report(Y_smote[test_index], Y_pred)
    recall = metrics.recall_score(Y_smote[test_index], Y_pred, labels=labels)
    score = metrics.accuracy_score(Y_smote[test_index], Y_pred)
    f1 = metrics.f1_score(Y_smote[test_index], Y_pred, labels=labels)
    result = {}
    result.__setitem__('confusionMatrix', conMat)
    result.__setitem__('report', report)
    result.__setitem__('Acc', score)
    result.__setitem__('recall', recall)
    result.__setitem__('f1', f1)
    scores.append(result)
    print('Fold: {} Class dist.: {}, Acc: {}'.format(k+1, np.bincount(Y_smote[train_index]), score))


for item in scores:
    print(item['confusionMatrix'])
    print(item['report'])
    print(item['Acc'])

del scores
