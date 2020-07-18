# -*- coding:of multinomial regression assignment -*-


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import linear_model
mdata = pd.read_csv("D:\\mdata.csv")
mdata.head(30)


mdata.describe()
mdata.id.value_counts()

sns.boxplot(x="id",y="female",data=mdata)
sns.boxplot(x="id",y="ses",data=mdata)
sns.boxplot(x="id",y="prog",data=mdata)
sns.boxplot(x="id",y="read",data=mdata)
sns.boxplot(x="id",y="write",data=mdata)
sns.boxplot(x="id",y="math",data=mdata)
sns.boxplot(x="id",y="science",data=mdata)
sns.boxplot(x="id",y="honors",data=mdata)

sns.stripplot(x="id",y="female",jitter=True,data=mdata)
sns.stripplot(x="id",y="ses",jitter=True,data=mdata)
sns.stripplot(x="id",y="prog",jitter=True,data=mdata)
sns.stripplot(x="id",y="read",jitter=True,data=mdata)
sns.stripplot(x="id",y="write",jitter=True,data=mdata)
sns.stripplot(x="id",y="math",jitter=True,data=mdata)
sns.stripplot(x="id",y="honors",jitter=True,data=mdata)

sns.pairplot(mdata,hue="id") 
sns.pairplot(mdata)

mdata.corr()
train,test = train_test_split(mdata,test_size = 0.2)
model = LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,1:],train.iloc[:,0])

train_predict = model.predict(train.iloc[:,1:])  
test_predict = model.predict(test.iloc[:,1:]) 
accuracy_score(train.iloc[:,0],train_predict)

accuracy_score(test.iloc[:,0],test_predict)
 
