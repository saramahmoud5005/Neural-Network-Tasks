# -*- coding: utf-8 -*-
"""Pengunis_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VE-HNZjnOkf6aAnukEBRKdJ-lmxZe09A

**Import libraries**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

"""**Load dataset**"""

def load_data():
  dataset=pd.read_csv('penguins.csv')
  return dataset

dataset=load_data()
dataset.head()

"""**Explore & Vizualise data**"""

dataset.info()

dataset.describe()

"""**Visualization**"""

def visualize(feature1,feature2,df):
     return sns.scatterplot(x=feature1, y=feature2, data=df, hue="species")
     #plt.scatter(x=feature1,y=feature2,data=df)
     #plt.show()

visualize('bill_length_mm','bill_depth_mm',dataset)

visualize('bill_length_mm','flipper_length_mm',dataset)

visualize('bill_length_mm','gender',dataset)

visualize('bill_length_mm','body_mass_g',dataset)

visualize('flipper_length_mm','bill_length_mm',dataset)

visualize('bill_depth_mm','gender',dataset)

visualize('bill_depth_mm','body_mass_g',dataset)

visualize('flipper_length_mm','gender',dataset)

visualize('flipper_length_mm','body_mass_g',dataset)

visualize('gender','body_mass_g',dataset)

"""**Preprocessing**"""

#Null handling,Categorical handling,Normalize ==> heba (prefare make it function)
df=dataset.copy()

def gender_encoding(g):
   gender_mapper={'male':1,'female':2}
   return gender_mapper[g]

def preprocess (x,y):
  le=preprocessing.LabelEncoder()
  mms = MinMaxScaler()
  for col in x:
    if x[col].dtypes== object:
      x[col].fillna(x[col].mode()[0],inplace=True)
      x[col]=x[col].apply(gender_encoding)
    x[col]=mms.fit_transform(x[col].values.reshape(-1,1))
  y=pd.DataFrame(le.fit_transform(y))
  return x,y

# x,y=preprocess(df.iloc[:,1:],df.iloc[:,0])
# x.head()

"""**train test split**


*   
30 train and 20 test per class


"""

def train_test_split(X,y):
  x_slice1=slice(0,30)
  x_slice2=slice(50,80)
  x_slice3=slice(30,50)
  x_slice4=slice(80,100)
  X_train=X[X.index.isin(np.r_[x_slice1, x_slice2])].to_numpy()
  y_train=y[y.index.isin(np.r_[x_slice1, x_slice2])].to_numpy()
  X_test=X[X.index.isin(np.r_[x_slice3, x_slice4])].to_numpy()
  y_test=y[y.index.isin(np.r_[x_slice3, x_slice4])].to_numpy()
  return X_train,y_train,X_test,y_test

# X_train,y_train,X_test,y_test=train_test_split(x,y)  
# print('X_train shape:'+str(X_train.shape))
# print('y_train shape:'+str(y_train.shape))
# print('X_test shape:'+str(X_test.shape))
# print('y_test shape:'+str(y_test.shape))
#X_test=np.insert(X_test, 0, np.ones(X_test.shape[0]),axis=1)

"""**Build model and train**"""

def intialize_parameters(is_bias,feature1,feature2,class1,class2,df):
  np.random.seed(1)
  if is_bias == True:
    W =np.random.randn(3,)
  else:
    W =np.random.randn(2,)
  data=df[['species',feature1,feature2]]
  data=data.loc[data['species'].isin([class1,class2])]
  return W,data

# W,data=intialize_parameters(True,'bill_length_mm','bill_depth_mm','Adelie','Gentoo',dataset)
# print(W.T)
# data.head()

def signum_function(a):
  return 1 if a > 0 else 0

def perceptron_learing(learning_rate,epochs,is_bias,feature1,feature2,class1,class2):
  ###load punguins.csv
  dataset=load_data()
  ###intialize param & prepare dataset
  W,dataset=intialize_parameters(is_bias,feature1,feature2,class1,class2,dataset)
  ###preprocess data
  X,y=preprocess (dataset.iloc[:,1:],dataset.iloc[:,0])
  #### split data into train and test 
  X_train,Y_train,X_test,Y_test=train_test_split(X,y)
  if is_bias == True:
    X_train=np.insert(X_train, 0, np.ones(X_train.shape[0]),axis=1)
    X_test=np.insert(X_test, 0, np.ones(X_test.shape[0]),axis=1)
  for e in range(epochs):
    for xi,yi in zip(X_train,Y_train):   
      a = np.dot(W.T,xi) 
      #print(W.shape)
      y = signum_function(a)
      if(y != yi):
        error = yi - y
        W = W + (learning_rate * error * xi)
   
  return W,X_train,Y_train,X_test,Y_test 

W,X_train,Y_train,X_test,Y_test=perceptron_learing(0.01,100,True,'bill_length_mm','bill_depth_mm','Adelie','Gentoo') 

print(W)
print('X_test shape:'+str(X_test.shape))
print('y_test shape:'+str(Y_test.shape))

"""**Test and evaluation**"""

def test(X_test,parameter):
  y_hat=np.dot(X_test,parameter.T)
  return y_hat

def evaluate(X_test,y_test,parameter):
  confusion_m=np.zeros((2,2))
  y_hat=test(X_test,parameter)
  for yi,y_hati in zip(y_test,y_hat):
    y_hati=signum_function(y_hati) 
    if (yi == y_hati):
      if yi==0:
        confusion_m[0][0]+=1
      else:
        confusion_m[1][1]+=1 
    else :
       if yi==0:
        confusion_m[0][1]+=1
       else:
        confusion_m[1][0]+=1 
  accuracy=(confusion_m[0][0]+confusion_m[1][1])/(confusion_m[0][0]+confusion_m[0][1]+confusion_m[1][0]+confusion_m[1][1])
  print("confusion matrix")
  print(confusion_m)
  print("accuracy: ",str(accuracy))
  return accuracy

#train accuracy
print('##################train accuracy##########################')
evaluate(X_train,Y_train,W)
print('##################test accuracy##########################')
evaluate(X_test,Y_test,W)

def plot_decision_boundry(X_test,y_test,parameters):
   feature_indx1,feature_index2,b,w1,w2=0,0,0,0,0
   if parameters.shape[0] < 3:
      feature_indx1=0
      feature_indx2=1
      b=0
      w1,w2=parameters[0],parameters[1]
   else:
      feature_indx1=1
      feature_indx2=2
      b,w1,w2= parameters[0],parameters[1],parameters[2]  
  # Plotting the decision boundary
   fig = plt.figure(figsize=(10,7))
   x_values = [np.min(X_test[:, feature_indx1] -5 ), np.max(X_test[:, feature_indx1] +5 )]
   # calcul y values
   y_values = np.dot((-1./w2), (np.dot(w1,x_values) + b))
   colors=['red' if l==0 else 'blue' for l in y_test]
   plt.scatter(X_test[:, feature_indx1], X_test[:, feature_indx2], color=colors)
   plt.plot(x_values, y_values, label='Decision Boundary')
   plt.show()


plot_decision_boundry(X_test,Y_test,W)