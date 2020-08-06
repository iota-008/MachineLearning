import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
labelencoder = preprocessing.LabelEncoder()
buying = labelencoder.fit_transform(list(data["buying"]))
maint = labelencoder.fit_transform(list(data["maint"]))
door = labelencoder.fit_transform(list(data["door"]))
persons = labelencoder.fit_transform(list(data["persons"]))
lug_boot = labelencoder.fit_transform(list(data["lug_boot"]))
safety = labelencoder.fit_transform(list(data["safety"]))
cls = labelencoder.fit_transform(list(data["class"]))

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2)

model = KNeighborClassifier(n_neighbor=5)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(y_predict)
