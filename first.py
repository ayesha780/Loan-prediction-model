# libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

# importing the file
df = pd.read_csv("filename")

#data cleaning
df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace = True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace = True)
df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean())
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df.isnull().sum()

#Visuals
x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y = df.iloc[:,12].values
print("Number of poeople who takes loan group by gender category")
print(df['Gender'].value_counts())
sns.countplot(x='Gender', data=df, palette='Set2')

print("Number of poeople who takes loan group by Marital status category")
print(df['Married'].value_counts())
sns.countplot(x='Married', data=df, palette='Set2')

print("Number of poeople who takes loan group by gender category")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df, palette = 'Set2')

print("Number of poeople who takes loan group by Employement category")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=df, palette='Set2')

print("Number of poeople who takes loan group by loan amount category")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount', data=df)

print("Number of poeople who takes loan group by credit history category")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=df , palette = 'Set2')

#training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder
LabelEncoder_x = LabelEncoder()

for i in range(0,5):
    X_train[:,i] = LabelEncoder_x.fit_transform(X_train[:,i])
    X_train[:,7] = LabelEncoder_x.fit_transform(X_train[:,7])

LabelEncoder_y = LabelEncoder()
y_train = LabelEncoder_y.fit_transform(y_train)
y_train

for i in range(0,5):
    X_test[:,i] = LabelEncoder_x.fit_transform(X_test[:,i])
    X_test[:,7] = LabelEncoder_x.fit_transform(X_test[:,7])

X_test

for i in range(0,5):
    X_test[:,i] = LabelEncoder_x.fit_transform(X_test[:,i])
    X_test[:,7] = LabelEncoder_x.fit_transform(X_test[:,7])

X_test   

#predictions

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
x_test = ss.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf_classifier  = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

from sklearn import metrics
y_pred = rf_classifier.predict(x_test)
print("acc of random forest clf is", metrics.accuracy_score(y_pred,y_test))
y_pred

from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,y_train)


