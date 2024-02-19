#importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#Reding the dataset.
df=pd.read_csv(r"C:\Users\Netha\Ananconda onlinclass\Mission learning\Decision Tree\car_evaluation.csv",header=None)
#print(df)
#print(df.shape)
#Changing the Headers for each column
col_names=['Buying','meant','doors','persons','lug_boot','safety','class']
df.columns=col_names
#print(col_names)
#print(df.head())
#print(df.info())
#checking the null values.
#print(df.isnull().sum())
#using the label encoder.
from sklearn.preprocessing import LabelEncoder
df[['Buying','meant','doors','persons','lug_boot',
    'safety','class']]=df[['Buying','meant','doors','persons','lug_boot',
    'safety','class']].apply(LabelEncoder().fit_transform)
#print(df.head())
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
#print(X.head())
#Training the model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#fitting to Decision Tree model.
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)
#print(tree.score(x_test,y_test)*100)
import matplotlib.pyplot as plt
# plt.scatter(y_pred,y_test)
# plt.plot(y_test,y_pred)
# plt.title('Car values prediction')
# plt.xlabel('x values')
# plt.ylabel('y values')
# plt.show()
plt.scatter(y_test,y_pred,color='red')
plt.plot(y_test,y_pred,color='blue')
plt.show()