#!/usr/bin/env python
# coding: utf-8

# **Import libraries**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


# **Load dataset**

# In[2]:


def load_data():
    dataset=pd.read_csv('penguins.csv')
    return dataset

dataset=load_data()
dataset.head()


# **Explore & Vizualise data**

# In[3]:


# dataset.info()


# In[4]:


dataset.describe()


# **Visualization**

# In[5]:


def visualize(feature1,feature2,df):
     sns.scatterplot(x=feature1, y=feature2, data=df, hue="species")


# In[6]:


visualize('bill_length_mm','bill_depth_mm',dataset)


# In[7]:


visualize('bill_length_mm','flipper_length_mm',dataset)


# In[8]:


visualize('bill_length_mm','gender',dataset)


# **Preprocessing**

# In[9]:


#Null handling,Categorical handling,Normalize ==> heba (prefare make it function)
df=dataset.copy()

def gender_encoding(g):
   gender_mapper={'male':1,'female':2}
   return gender_mapper[g]

def preprocess (x,y):
  one_hot_encoder=preprocessing.OneHotEncoder()
  mms = MinMaxScaler()
  for col in x:
    if x[col].dtypes== object:
      x[col].fillna(x[col].mode()[0],inplace=True)
      x[col]=x[col].apply(gender_encoding)
    x[col]=mms.fit_transform(x[col].values.reshape(-1,1))
  species_array = one_hot_encoder.fit_transform(y.values.reshape(-1,1)).toarray()
  species_labels = np.array(one_hot_encoder.categories_).ravel()#to make it an array, and .ravel() to convert it from array of arrays to array of strings
  y = pd.DataFrame(species_array, columns=species_labels)  
  return x,y

#x,y=preprocess(df.iloc[:,1:],df.iloc[:,0])
# x.head()


# **train test split**
# 
# 
# *   
# 30 train and 20 test per class
# 
# 

# In[10]:


def train_test_split(X,y):
    x_slice1=slice(0,30)
    x_slice2=slice(50,80)
    x_slice3=slice(100,130)

    x_slice4=slice(30,50)
    x_slice5=slice(80,100)
    x_slice6=slice(130,150)

    X_train=X[X.index.isin(np.r_[x_slice1, x_slice2 , x_slice3])].to_numpy()
    y_train=y[y.index.isin(np.r_[x_slice1, x_slice2, x_slice3])].to_numpy()

    X_test=X[X.index.isin(np.r_[x_slice4, x_slice5 , x_slice6])].to_numpy()
    y_test=y[y.index.isin(np.r_[x_slice4, x_slice5 , x_slice6])].to_numpy()

    return X_train,y_train,X_test,y_test

# X_train,y_train,X_test,y_test=train_test_split(x,y)  
# print('X_train shape:'+str(X_train.shape))
# print('y_train shape:'+str(y_train.shape))
# print('X_test shape:'+str(X_test.shape))
# print('y_test shape:'+str(y_test.shape))


# **Build model and train**

# In[11]:


def intialize_parameters(is_bias,layer_dims):
    np.random.seed(3)
    print(len(layer_dims))
    parameters = {}
    for l in range(1, (len(layer_dims))):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        if is_bias:
            parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1)
        else:
            parameters['b' + str(l)] =np.zeros((layer_dims[l], 1)) 
    return parameters

# parameters = intialize_parameters(True,[5,4,3])

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# In[12]:


#Activation Functions

def sigmoid(z):
    return 1/(1+np.exp(-z))


def hyperbolicTangent(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
   


def linear_activation_forward(prevAct , W , b , activationFn):

    if activationFn == "sigmoid":

        Z = np.dot(W,prevAct) + b
        linearCache = (prevAct,W,b)
        A  = sigmoid(Z)

    elif activationFn == "hyperbolicTangent":

        Z = np.dot(W,prevAct) + b
        linearCache = (prevAct,W,b)
        A  = hyperbolicTangent(Z)

    return A , linearCache


# In[13]:


def forward_prop(A,parameters,activation_fn):
  
    caches = []
    L= len(parameters) // 2
  ##print(L)
    for l in range(1,L):
        prevAct = A
        A,cache = linear_activation_forward(prevAct , parameters['W' +str(l)] , parameters['b'+str(l)],activation_fn)
        caches.append(cache)

    Al, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation_fn)
    caches.append(cache)    


    return Al,caches #Al == output of last layer


# In[14]:


def backward_prop(Al, Y, caches, activation_fn):
  # cashes --> list of cashe
  # cashe --> tuple(A_prev, W, b)

    GDs = {}
    L = len(caches) # num of layers 
  
  #Output layer
    if activation_fn == "sigmoid":
        GD = np.dot(np.dot((Y-Al), np.transpose(Al)), (1-Al)) # GD size == Y or Al size
    elif activation_fn == "hyperbolicTangent":
        GD = np.dot(np.dot((Y-Al), np.transpose(1-Al)), (1+Al)) # GD size == Y or Al size
    GDs["dw"+str(L)] = GD
  
  #Hidden layers
    for i in reversed(range(1, L)):
        if activation_fn == "sigmoid":
            dw = np.dot((caches[i][0]), (1-caches[i][0]).transpose())
        elif activation_fn == "hyperbolicTangent":
             dw = np.dot((1-caches[i][0]), (1+caches[i][0]).transpose())   ##1 - pow (caches[i][0],2)
        segma_times_W = np.dot(GDs["dw"+str(i+1)].transpose(), caches[i][1])
 
        GD = np.dot(dw, segma_times_W.transpose()) # GD size == A size in the cache
        GDs["dw"+str(i)] = GD 
    

    return GDs


# Testing backward_prop function

# In[15]:


# caches = []

# W1 = np.random.randn(4,3)*0.01
# b1 = np.zeros((4,1))
# A1 = np.random.randn(3,1)*0.01
# cache = (A1, W1, b1)
# caches.append(cache)

# W2 = np.random.randn(3,4)*0.01
# b2 = np.zeros((3,1))
# A2 = np.random.randn(4,1)*0.01
# cache = (A2, W2, b2)
# caches.append(cache)

# W3 = np.random.randn(2,3)*0.01
# b3 = np.zeros((2,1))
# A3 = np.random.randn(3,1)*0.01
# cache = (A3, W3, b3)
# caches.append(cache)

# Al = np.random.randn(2,1)*0.01
# Y = np.random.randn(2,1)*0.01

# # Testing Sigmoid Function
# GDs = backward_prop(Al,Y,caches,"sigmoid")

# print("GDs[W1]: ")
# print(GDs["dw1"])
# print("GDs[W2]: ")
# print(GDs["dw2"])
# print("GDs[W3]: ")
# print(GDs["dw3"])

# print("####################################")
# #Testing Hyperbolic Tangent Function
# GDs = backward_prop(Al,Y,caches,"hyperbolicTangent")
# print("GDs[W1]: ")
# print(GDs["dw1"])
# print("GDs[W2]: ")
# print(GDs["dw2"])
# print("GDs[W3]: ")
# print(GDs["dw3"])


# In[16]:


def update_parameters(parameters,grads,learning_rate,caches):
    L = len(caches)
    for l in range(L):
            A_prev,W,b=caches[l]
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] +learning_rate * np.dot(grads["dw" + str(l+1)],A_prev.reshape(-1,1).T)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] +learning_rate * np.multiply(grads["dw" + str(l+1)],b)
         
    return parameters


# In[17]:


def NN_model(lr,ephocs,layer_dims,is_bias,activation_fn):
    ###load punguins.csv
    dataset=load_data()
    ###preprocess data
    X,y=preprocess (dataset.iloc[:,1:],dataset.iloc[:,0])
    #### split data into train and test 
    X_train,Y_train,X_test,Y_test=train_test_split(X,y)

    #initialize parameters
    parameters=intialize_parameters(is_bias,layer_dims)
    
    # Loop (gradient descent)
    for i in range(0, ephocs):

        for xi,yi in zip(X_train,Y_train): 

             # Forward propagation
        
             AL , caches=forward_prop(xi.T.reshape(-1,1),parameters,activation_fn)
        # Backward propagation.
             grads =backward_prop(AL, yi.T.reshape(-1,1), caches, activation_fn)
        # Update parameters.
             parameters =update_parameters(parameters, grads, lr,caches)
  
    return parameters,X_train,Y_train,X_test,Y_test

# param,X_train,Y_train,X_test,Y_test=NN_model(0.01,10,[5,4,4,3],True,'sigmoid')  
# print("W1 = " + str(param["W1"]))
# print("b1 = " + str(param["b1"]))
# print("W2 = " + str(param["W2"]))
# print("b2 = " + str(param["b2"]))
# print("W3 = " + str(param["W3"]))
# print("b3 = " + str(param["b3"]))


# In[18]:


def encoding_output(ar):
    max =0.0
    max_indx =-1
  
    for i,x in enumerate(ar):
        if(x > max):
            max=x
            max_indx=i
    return max_indx

def test(X_test,parameters,activation_fn):
    Al , caches=forward_prop(X_test.T.reshape(-1,1),parameters,activation_fn)
    res=encoding_output(Al)
    return res


def evaluate(X_test,y_test,parameter,activation_fn):
    confusion_m=np.zeros((3,3))
    for yi,xi in zip(y_test,X_test):
        y_hati=test(xi,parameter,activation_fn)
        yi= encoding_output(yi)
        if (yi == y_hati):
            confusion_m[yi][yi]+=1
        else :
            confusion_m[yi][y_hati]+=1

    accuracy=(confusion_m[0][0]+confusion_m[1][1]+confusion_m[2][2])/(confusion_m[0][0]+confusion_m[0][1]+confusion_m[0][2]+confusion_m[1][0]+confusion_m[1][1]+confusion_m[1][2]+confusion_m[2][0]+confusion_m[2][1]+confusion_m[2][2])
    print("confusion matrix")
    print(confusion_m)
    print("accuracy: ",str(accuracy)) 
    return accuracy           


# In[19]:


#test model
# parameters,X_train,Y_train,X_test,Y_test=NN_model(0.01,1000,[5,4,3],True,"hyperbolicTangent")
# evaluate(X_train,Y_train,parameters,"hyperbolicTangent")


# In[ ]:




