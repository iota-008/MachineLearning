import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from pandas_profiling import ProfileReport


# d1 = pd.read_csv("student-mat.csv", sep=":")
# d2 = pd.read_csv("student-por.csv", sep=";")

# df = d1.append(d2)


df = pd.read_csv("student-mat.csv", sep=";")
print(df.head())

# profile = ProfileReport(
#     df, title="Profile Report of the students grade  Dataset", explorative=True
# )
# profile.to_file("StudentsGrades.html")

df = df[["failures", "Medu", "Fedu", "Dalc",
         "Walc", "absences", "G1", "G2", "G3"]]

# pd.Series(np.where(df.activities.values == 'yes', 1, 0),df.index)
# df.activities.map(dict(yes=1, no=0))
# df["activities"] = df.activities.map({'yes': 1, 'no': 0})
print(df.head())

predict = "G3"
x = np.array(
    df.drop([predict], axis=1))
y = np.array(df[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2)

reg = linear_model.LinearRegression(normalize=True)
reg.fit(x_train, y_train)
accuracy = reg.score(x_test, y_test)
print("accuracy score = ", accuracy)
