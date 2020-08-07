import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

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

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(y_predict)

# print("accuracy = ", accuracy)

best = 0
for i in range(100):
    accuracy = model.score(x_test, y_test)
    if accuracy > best:
        best = accuracy
        with open("cars.pickle", "wb") as f:
            pickle.dump(model, f)

better_model = open("cars.pickle", "rb")
best_model = pickle.loads(best_model)
best_model.predict(x_test)
print("accuracy = ", accuracy)
