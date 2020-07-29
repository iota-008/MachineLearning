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
# age+Fedu+Medu+Fjob+Mjob+famsize+sex+Pstatus + \absences+famsize+Dalc+famrel+traveltime


# df = df[["failures", "Medu", "Fedu", "Dalc","Walc", "absences", "G1", "G2", "G3"]]

df = df[["G1", "G2", "G3", "age", "Fedu", "Medu", "Fjob", "Mjob", "famsize", "sex",
         "Pstatus", "absences", "Dalc", "famrel", "traveltime"]]

dummy_fjob = pd.get_dummies(df['Fjob'])
dummy_mjob = pd.get_dummies(df['Mjob'])
dummy_famsize = pd.get_dummies(df['famsize'])
dummy_sex = pd.get_dummies(df['sex'])
dummy_pstatus = pd.get_dummies(df['Pstatus'])

df = df.merge(dummy_fjob, left_index=True, right_index=True)
df = df.merge(dummy_mjob, left_index=True, right_index=True)
df = df.merge(dummy_famsize, left_index=True, right_index=True)
df = df.merge(dummy_sex, left_index=True, right_index=True)
df = df.merge(dummy_pstatus, left_index=True, right_index=True)

df.drop(['Fjob', 'Mjob', 'famsize', 'sex', 'Pstatus'], axis=1, inplace=True)
# pd.Series(np.where(df.activities.values == 'yes', 1, 0),df.index)
# df.activities.map(dict(yes=1, no=0))
# df["activities"] = df.activities.map({'yes': 1, 'no': 0})
print(df.head())

predict = "G3"
x = np.array(
    df.drop([predict], axis=1))
y = np.array(df[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

linear = linear_model.LinearRegression(normalize=True)
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print("accuracy score = ", accuracy)

print("coeffcient", linear.coef_)
print("intercept", linear.intercept_)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print("predicted: ", round(predictions[i]), "actual value: ", y_test[i])
