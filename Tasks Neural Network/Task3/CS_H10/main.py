#!/usr/bin/env python
# coding: utf-8

# In[9]:


from tkinter import *
from tkinter import ttk


# In[10]:


pip install import-ipynb


# In[11]:


# Create a GUI app
import import_ipynb
import Task3_NN
App = Tk()
App.title("Task3")

# Set the geometry of the app
App.geometry("430x500")

dictionary = {}
weight = 2
num_bias = 2
res = list()


# In[12]:


def getSizeHiddenLayers():
    listOfSizes=dictionary['sizesOfHiddenLayers'].split(",")
    print(listOfSizes)
    for i in range(len(listOfSizes)):
        print(listOfSizes[i])
        listOfSizes[i] = int(listOfSizes[i])
    listOfSizes.insert(0,5)
    listOfSizes.append(3)
    dictionary['sizesOfHiddenLayers_Int']=listOfSizes


# In[13]:


def getValues():
    print("yes ")
    dictionary['ActivationFunction'] = ActivationFunction.get()
    dictionary['LearningRate'] = learningRate.get()
    dictionary['Epochs'] = epochs.get()
    dictionary['NumHiddenLayers'] = hiddenLayer.get()
    dictionary['Bias'] = bias.get()
    dictionary['sizesOfHiddenLayers']=numberOfNeurons.get()

    getSizeHiddenLayers()

    print("Activation Function = ", dictionary['ActivationFunction'])
    print("LearningRate = ", dictionary['LearningRate'])
    print("Epochs = ", dictionary['Epochs'])
    print("NumHiddenLayers = ", dictionary['NumHiddenLayers'])
    print("Bias : ", dictionary['Bias'])
    print("sizesOfHiddenLayers : ", dictionary['sizesOfHiddenLayers'])
    print("sizesOfHiddenLayers_Int : ", dictionary['sizesOfHiddenLayers_Int'])


# In[14]:


def Train():
    print("Train")
    getValues()
    parameters,X_train,Y_train,X_test,Y_test=Task3_NN.NN_model(dictionary['LearningRate'],dictionary['Epochs'],dictionary['sizesOfHiddenLayers_Int']
                      ,dictionary['Bias'],dictionary['ActivationFunction'])

    res.append(parameters)
    res.append(X_train)
    res.append(Y_train)
    res.append(X_test)
    res.append(Y_test)
    acc=Task3_NN.evaluate(res[1],res[2],res[0],dictionary['ActivationFunction'])
    accuracy_train(acc)
    
    

def accuracy_train(acc):
    dictionary['Accuracy_train'] = acc
    label_Accuracy2 = Label(App, text="Accuracy Train = " + str(acc), font=25)
    label_Accuracy2.grid(row=22, column=1, padx=5, pady=5)    


# In[15]:


def Test():
    print("Test")
    acc=Task3_NN.evaluate(res[3],res[4],res[0],dictionary['ActivationFunction'])
    accuracy_test(acc)


def accuracy_test(acc):
    dictionary['Accuracy_test'] = acc
    #print("accu test = "+dictionary['Accuracy_test'])
    label_Accuracy1 = Label(App, text="Accuracy Testing = " + str(acc), font=25)
    label_Accuracy1.grid(row=24, column=1, padx=5, pady=5)


# In[16]:


# Define Activation Functions
activationFunctions = ('sigmoid', 'hyperbolicTangent')

# Create a label and Combobox for class1
ActivationFunction = StringVar()

activationFunction_label = Label(App, text="Activation Function", width=20, font=25)
activationFunction_label.grid(row=3, column=1, padx=10, pady=10)

combo_activationFunction = ttk.Combobox(App, textvariable=ActivationFunction)
combo_activationFunction['values'] = activationFunctions
combo_activationFunction['state'] = 'readonly'
combo_activationFunction.grid(row=3, column=2, padx=10, pady=10)

# label and entry for learning rate
learningRate = DoubleVar()

label_learningRate = Label(App, text="Learning Rate", font=20)
label_learningRate.grid(row=5, column=1, padx=10, pady=10)

entry_learningRate = Entry(App, font=20, textvariable=learningRate, width=20)
entry_learningRate.grid(row=5, column=2)

# label and entry for epochs
epochs = IntVar()

label_epochs = Label(App, text="Epochs", font=25)
label_epochs.grid(row=6, column=1)

entry_epochs = Entry(App, font=20, textvariable=epochs, width=20)
entry_epochs.grid(row=6, column=2)

# label and entry for mse
hiddenLayer = IntVar()

label_hiddenLayer = Label(App, text="Layers", font=25)
label_hiddenLayer.grid(row=8, column=1)

entry_hiddenLayer = Entry(App, font=20, textvariable=hiddenLayer, width=20)
entry_hiddenLayer.grid(row=8, column=2)

# label and entry for mse
numberOfNeurons = StringVar()

label_numberOfNeurons = Label(App, text="Neurons", font=25)
label_numberOfNeurons.grid(row=9, column=1)

entry_numberOfNeurons = Entry(App, font=20, textvariable=numberOfNeurons, width=20)
entry_numberOfNeurons.grid(row=9, column=2)


# label and Radiobutton for bias
bias = BooleanVar(value=False)

label_bias = Label(App, text="Bias", font=25)
label_bias.grid(row=13, column=1, padx=5, pady=5)

Radiobutton_bias = Radiobutton(App, text="YES", variable=bias, value=True)
Radiobutton_bias.grid(row=14, column=1, padx=2)

Radiobutton_bias = Radiobutton(App, text="NO", variable=bias, value=False)  # ,command=getValues())
Radiobutton_bias.grid(row=15, column=1, padx=2)

# Create a button to Train
button_Train = Button(App, text="Train", width=20, command=Train, font=15, pady=5)
button_Train.grid(row=16, column=1)

# Create a button to Test
button_Test = Button(App, text="Test", width=20, command=Test, font=15, pady=5)
button_Test.grid(row=16, column=2)


App.mainloop()


# In[ ]:





# In[ ]:




