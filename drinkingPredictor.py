import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

math_student_data = pd.read_csv('student-mat.csv') #read data into a pandas data frame

#remove columns which are not going to be used from the data
del math_student_data['school']
del math_student_data['reason']
del math_student_data['schoolsup']
del math_student_data['famsup']
del math_student_data['paid']
del math_student_data['health']
del math_student_data['G1']
del math_student_data['G2']
del math_student_data['Mjob']
del math_student_data['Fjob']
del math_student_data['guardian']
print('Data Features: ')
print(math_student_data.loc[0])
print('')
math_student_data = math_student_data.to_numpy() #convert the pandas dataframe to a numpy array

def convert_to_binary_representation(list, indexList, firstLabelList):
    listCopy = list.copy()
    for item in listCopy:
        labelCount = 0
        for index in indexList:
            if item[index] == firstLabelList[labelCount]:
                item[index] = 0
            else:
                item[index] = 1
            labelCount += 1
    return listCopy

#converts the features which are binary to the numbers 0 and 1
math_student_data = convert_to_binary_representation(math_student_data, (0, 2, 3, 4, 10, 11, 12, 13, 14), ('F', 'U', 'LE3', 'T', 'no', 'no', 'no', 'no', 'no'))

print(math_student_data[0:15])
print(' ')

def generate_features_and_targets(data): #split the data into features with all the characteristics of each student were measuring, and targets: their dalc + walc
    features = []
    targets = []
    for student in data:
        #combine workday and weekend alcohol consumption into one figure
        student = list(student)
        dalc = student.pop(18)
        walc = student.pop(18)
        features.append(student)
        targets.append(dalc + walc)

    return np.array(features), np.array(targets)

features, targets = generate_features_and_targets(math_student_data)

xTrain, xTest, yTrain, yTest = train_test_split(features, targets) # random_state=7

#all of the following models were tried and the one that preforms best, the Random Forest decision tree, was left uncommented
#model = KNeighborsRegressor(n_neighbors=2)
#model = LinearRegression()

#model =
#model = MLPRegressor(solver='lbfgs', random_state=2, hidden_layer_sizes=[100, 100])
model = Lasso(alpha=.03, max_iter=1000)
print(model.get_params())
model = Pipeline([("scaler", MinMaxScaler()), ("model",  model)])


''' This code was used to select the best parameters and model.'''
'''
param_grid = [{'model__n_estimators': [5, 10, 50, 100, 150, 250, 500]}, {'model__alpha': [.001, .01, .1, 1, 10, 100, 1000]}, {'model__alpha': [.001, .01, .1, 1, 10, 100, 1000], 'model__max_iter': [1000, 5000, 10000, 50000, 100000, 500000]}, {'model__C': [.001, .01, .1, 1, 10, 100, 1000], 'model__gamma': [.001, .01, .1, 1, 10, 100, 1000]}]
models = [RandomForestRegressor(n_estimators=500), Ridge(), Lasso(), SVR()] #, random_state=9
for i in range(len(models)):
    model = Pipeline([("scaler", MinMaxScaler()), ("model",  models[i])])
    grid_search = GridSearchCV(model, param_grid[i], cv=20, return_train_score=True)
    model = grid_search.fit(xTrain, yTrain)
    print(model.best_params_)
    print(model.score(xTest, yTest))
    print('')

#print(cross_val_score(model, features, targets).mean()) #cross-val scores were compared between all the model choices to select one
'''

model.fit(xTrain, yTrain)
predictions = model.predict(xTest)
actual = yTest

'''
for prediction in range(len(predictions)):
    print('Predicted: ' + str(predictions[prediction]))
    print('Actual: ' + str(actual[prediction]))
    print(' ')
'''

#print(model.score(xTrain, yTrain))
#print(model.score(xTest, yTest))
print('')
print("Model Coefficients:")
print(model['model'].coef_)
print('')
print("Cross Validation Accuracy: " + str(cross_val_score(model, features, targets).mean()))