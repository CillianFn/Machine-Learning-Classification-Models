import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Read in the tab seperated text file using Pandas' read_csv method
patients = pd.read_csv('autoimmune.txt', sep="\t", header=None)

#Transpose the data so that patients are in rows
patients = patients.T 

#Label the columns, allows for supervised learning
patients.columns = ['Age', 'Blood_Pressure', 'BMI', 'Plasma_level', 'Autoimmune_Disease', 
                    'Adverse_events', 'Drug_in_serum', 'Liver_function', 'Activity_test', 'Secondary_test']

#Converting target values from string to int
patients.Autoimmune_Disease.replace(['positive', 'negative'], [1, 0], inplace=True)  


print(patients)

#Remove target values from X, include all rows 
X = patients.loc[:, patients.columns != 'Autoimmune_Disease']

#Specify target values 
y = patients.Autoimmune_Disease


#Declare decision tree and fit the model 
dt = tree.DecisionTreeClassifier(max_depth=3)
dt.fit(X, y)

#Predict future performance of Decision tree model using KFold cross validation (k=10)
performance_dt = cross_val_score(dt, X, y, cv=10) 

#Print the average predicted performance from the 10 folds
print('\nDecision Tree')
print(performance_dt.mean()) 


#Declare decision tree and fit the model 
logR = LogisticRegression()
logR.fit(X, y)


performance_logR = cross_val_score(logR, X, y, cv=10) 
print('\nLogistic Regression')
print(performance_logR.mean()) 
