import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model


d1 = pd.read_csv("student-mat.csv", sep=":")
d2 = pd.read_csv("student-por.csv", sep=";")

df = d1.append(d2)


df = pd.read_csv("student-mat.csv", sep=";")
print(df.head())

predict = "G3"
x = np.array(df.drop([predict], axis=1))
y = np.array(df[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2)
