import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from pandas_profiling import ProfileReport
import statsmodels.api as sm
from patsy import dmatrices
import matplotlib.pyplot as plt
import pickle


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
print(df.columns)

predict = "G3"
x = np.array(
    df.drop([predict], axis=1))
y = np.array(df[predict])

# testSize = [0.1, 0.2, 0.3, 0.4]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2)
# best = 0
# for _ in range(1000):

#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
#         x, y, test_size=0.2)

#     linear = linear_model.LinearRegression(normalize=True)
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     print("accuracy score = ", accuracy)
#     if accuracy > best:
#         best = accuracy
#         with open("predicting_students_grades.pickle", "wb") as f:
#             pickle.dump(linear, f)


# # accuracy = linear.score(x_test, y_test)
# print("best accuracy = ", best)

pickle_file = open("predicting_students_grades.pickle", "rb")
linear = pickle.load(pickle_file)


print("coeffcient", linear.coef_)
print("intercept", linear.intercept_)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print("predicted: ", round(predictions[i]), "actual value: ", y_test[i])


# Create the training and testing data sets.
# mask = np.random.rand(len(df)) < 0.8
# df_train = df[mask]
# df_test = df[~mask]
# print('Training data set length='+str(len(df_train)))
# print('Testing data set length='+str(len(df_test)))

# # Setup the regression expression in patsy notation. We are telling patsy that BB_COUNT is our dependent variable and
# # it depends on the regression variables: DAY, DAY_OF_WEEK, MONTH, HIGH_T, LOW_T and PRECIP.
# expr = """G3 ~ G1  + G2 + age + famrel + Fedu + Medu + absences + Dalc + traveltime + at_home_x + at_home_y + health_x + health_y + other_x + other_y + services_x + services_y + teacher_x + teacher_y + GT3 + LE3 + F + M + A + T """

# # Set up the X and y matrices
# y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
# y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

# # Using the statsmodels GLM class, train the Poisson regression model on the training data set.
# poisson_training_results = sm.GLM(
#     y_train, X_train, family=sm.families.Poisson()).fit()

# # Print the training summary.
# print(poisson_training_results.summary())

# # Make some predictions on the test data set.
# poisson_predictions = poisson_training_results.get_prediction(X_test)

# # .summary_frame() returns a pandas DataFrame
# predictions_summary_frame = poisson_predictions.summary_frame()
# print(predictions_summary_frame)

# predicted_counts = predictions_summary_frame['mean']
# actual_counts = y_test['G3']


# # Mlot the predicted counts versus the actual counts for the test data.
# fig = plt.figure()
# fig.suptitle('Predicted versus actualfinal grades of students')
# predicted, = plt.plot(X_test.index, predicted_counts,
#                       'bo-', label='Predicted counts')
# actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
# plt.legend(handles=[predicted, actual])
# plt.show()

# # Show scatter plot of Actual versus Predicted counts
# plt.clf()
# fig = plt.figure()
# fig.suptitle('Scatter plot of Actual versus Predicted counts')
# plt.scatter(x=predicted_counts, y=actual_counts, marker='.')
# plt.xlabel('Predicted counts')
# plt.ylabel('Actual counts')
# plt.show()
