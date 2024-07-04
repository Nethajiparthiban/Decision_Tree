import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\Netha\Ananconda_onlineclass\Mission learning\ML from codebasics\Machine Learning\Decision_Tree\titanic.csv")
#print(df.columns)
#print(df.head())
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin', 'Embarked'],axis='columns',inplace=True)
#print(df.head())
df.isnull().sum()
mean=np.floor(df['Age'].mean())
df['Age']=df['Age'].fillna(np.floor(df['Age'].mean()))
#print(df.isnull().sum())
#print(df.head())
#As we saw in the output Sex columns should be replace with numerical value we use label encoder
from sklearn.preprocessing import LabelEncoder
age_en=LabelEncoder()
df['Sex']=age_en.fit_transform(df['Sex'])
#print(df.head())
features=df.drop(['Survived'],axis='columns')
target=df.Survived
#The data set little lengthy so we use train split method
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=0)
# print(len(x_train))
#print(y_test)
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
import pickle
with open('titanic_clf','wb') as f:
    pickle.dump(clf,f)
with open('titanic_clf','rb') as k:
    new_clf=pickle.load(k)
print(clf.predict([[3,1,29.0,14.4583]]))#0