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
