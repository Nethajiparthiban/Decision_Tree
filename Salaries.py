import pandas as pd
df=pd.read_csv(r"C:\Users\Netha\Ananconda_onlineclass\Mission learning\ML from codebasics\Machine Learning\Decision_Tree\salaries.csv")
#print(df.head())
#As we saw the output we need convert 3 columns company,job,degree
# we will use Label encoder method
from sklearn.preprocessing import LabelEncoder
#we have 3 different columns so we use 3 encoders with different names
company_en=LabelEncoder()
job_en=LabelEncoder()
degree_en=LabelEncoder()
df['company']=company_en.fit_transform(df.company)
df['job']=job_en.fit_transform(df.job)
df['degree']=degree_en.fit_transform(df.degree)
#print(df)
feature=df.drop(['salary_more_then_100k'],axis='columns')
target=df.salary_more_then_100k
#print(target)
#now it is time to fit to the tree algoritham
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(feature,target)
print(clf.score(feature,target))
#print(df)
print(clf.predict([[2,2,0]]))#0
print(clf.predict([[2,1,1]]))#1
import pickle
with open('new_pic','wb') as f:
    pickle.dump(clf,f)
with open('new_pic','rb') as k:
    new=pickle.load(k)

print(new.predict([[2,1,0]]))
