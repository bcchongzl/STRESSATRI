#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
 
from sklearn import preprocessing
import warnings
from joblib import dump, load
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegressionCV

train = pd.read_csv("train.csv")
train = train.loc[train['condition'] != 'interruption']


conditions = dict(train['condition'].value_counts())
labels = list(conditions.keys())
counts = list(conditions.values())
plt.figure()
plt.bar(labels,counts, color ='green',width = 0.4)
plt.savefig('condition.png')

 
train['condition'] =  train['condition'].map({'no stress':0,'time pressure':1})



reduced_train = train[['MEAN_RR',	'SDRR_RMSSD_REL_RR'	,	'VLF_PCT'	,'condition']]
 
reduced_train['ST'] = pd.cut( reduced_train.SDRR_RMSSD_REL_RR, [0, 1, 2, 3, 4], labels=['1', '2', '3', '4' ])

category_feature=pd.get_dummies(reduced_train['ST'],)
reduced_train=pd.concat([reduced_train,category_feature],axis=1)
 

data = reduced_train[['MEAN_RR','VLF_PCT', '1','2','3','4','condition']]
data[ 'MEAN_RR' ] = (data[ 'MEAN_RR' ]- data[ 'MEAN_RR' ].min())/(data[ 'MEAN_RR' ].max()-data[ 'MEAN_RR' ].min())
data[ 'VLF_PCT' ] = (data[ 'VLF_PCT' ]- data[ 'VLF_PCT' ].min())/(data[ 'VLF_PCT' ].max()-data[ 'VLF_PCT' ].min())
data[ 'HR' ] = data[ 'MEAN_RR' ]*(120-50) + 50
data[ 'age' ] = data[ 'MEAN_RR' ]*(110-10) + 10
data = data[['HR','age', '1','2','3','4','condition']]
data[ 'age' ] =data[ 'age' ].map(int)
data.to_csv("data.csv",index=None)

x_train = data.iloc[:,:-1]
y_train = data.iloc[:,-1]
 
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=1, test_size=0.2)

model = LogisticRegressionCV(  cv=3)
model.fit(x_train,y_train)


model = DecisionTreeClassifier(criterion="entropy", max_depth=14)
model.fit(x_train,y_train)




prediction = model.predict(x_test)
print(classification_report(y_test,prediction))
dump(model, 'model.pkl')











