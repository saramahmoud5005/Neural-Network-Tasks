from tkinter import *
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import adaline_penguins_classififcation

import matplotlib.pyplot as plt
from tkinter import messagebox

# Create a GUI app
App = Tk()
App.title("Task2")

# Set the geometry of the app
App.geometry("550x550")

dictionary = {}
weight = 2
num_bias = 2
res = list()


# Function to clear the form

def getValues():
    dictionary['Feature1'] = Feature1.get()
    dictionary['Feature2'] = Feature2.get()
    dictionary['Class1'] = Class1.get()
    dictionary['Class2'] = Class2.get()
    dictionary['LearningRate'] = learningRate.get()
    dictionary['Epochs'] = epochs.get()
    dictionary['MSE'] = mse.get()
    dictionary['Bias'] = bias.get()


def Train():
    print("Train")
    getValues()
    W, x_train, y_train, x_test, y_test = adaline_penguins_classififcation.adaline_learing(dictionary['LearningRate'],
                                                                                     dictionary['MSE'],
                                                                                     dictionary['Epochs'],
                                                                                     dictionary['Bias'],
                                                                                     dictionary['Feature1'],
                                                                                     dictionary['Feature2'],
                                                                                     dictionary['Class1'],
                                                                                     dictionary['Class2'])

    res.append(W)
    res.append(x_train)
    res.append(y_train)
    res.append(x_test)
    res.append(y_test)
    acc = adaline_penguins_classififcation.evaluate(res[1], res[2], res[0])
    accuracy_train(acc)


def Test():
    print("Test")
    acc = adaline_penguins_classififcation.evaluate(res[3], res[4], res[0])
    accuracy_test(acc)


def visualize():
    print("visualize")
    getValues()
    dataset = adaline_penguins_classififcation.load_data()
    df = dataset.copy()
    adaline_penguins_classififcation.visualize(dictionary['Feature1'], dictionary['Feature2'], df)


def accuracy_test(acc):
    dictionary['Accuracy_test'] = acc
    #print("accu test = "+dictionary['Accuracy_test'])
    label_Accuracy1 = Label(App, text="Accuracy Testing = " + str(acc), font=25)
    label_Accuracy1.grid(row=16, column=1, padx=5, pady=5)


def accuracy_train(acc):
    dictionary['Accuracy_train'] = acc
    label_Accuracy2 = Label(App, text="Accuracy Train = " + str(acc), font=25)
    label_Accuracy2.grid(row=15, column=1, padx=5, pady=5)


# Define features
features = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')

# Create a label and Combobox for feature1
Feature1 = StringVar()

feature1_label = Label(App, text="Feature 1", width=10, font=25)
feature1_label.grid(row=1, column=1, padx=10, pady=10)

combo_feature1 = ttk.Combobox(App, textvariable=Feature1)
combo_feature1['values'] = features
combo_feature1['state'] = 'readonly'
combo_feature1.grid(row=1, column=2, padx=10, pady=10)

# Create a label and Combobox for feature2
Feature2 = StringVar()

feature2_label = Label(App, text="Feature 2", width=10, font=25)
feature2_label.grid(row=2, column=1, padx=10, pady=10)

combo_feature2 = ttk.Combobox(App, textvariable=Feature2)
combo_feature2['values'] = features
combo_feature2['state'] = 'readonly'
combo_feature2.grid(row=2, column=2, padx=10, pady=10)

# Define classes
classes = ('Adelie', 'Gentoo', 'Chinstrap')

# Create a label and Combobox for class1
Class1 = StringVar()

class1_label = Label(App, text="Class 1", width=10, font=25)
class1_label.grid(row=3, column=1, padx=10, pady=10)

combo_class1 = ttk.Combobox(App, textvariable=Class1)
combo_class1['values'] = classes
combo_class1['state'] = 'readonly'
combo_class1.grid(row=3, column=2, padx=10, pady=10)

# Create a label and Combobox for class2
Class2 = StringVar()

class2_label = Label(App, text="Class 2", width=10, font=25)
class2_label.grid(row=4, column=1, padx=10, pady=10)

combo_class2 = ttk.Combobox(App, textvariable=Class2)
combo_class2['values'] = classes
combo_class2['state'] = 'readonly'
combo_class2.grid(row=4, column=2, padx=10, pady=10)

# label and entry for learning rate
learningRate = DoubleVar()

label_learningRate = Label(App, text="Learning Rate", font=20)
label_learningRate.grid(row=5, column=1, padx=10, pady=10)

entry_learningRate = Entry(App, font=20, textvariable=learningRate, width=16)
entry_learningRate.grid(row=5, column=2)

# label and entry for epochs
epochs = IntVar()

label_epochs = Label(App, text="Epochs", font=25)
label_epochs.grid(row=6, column=1)

entry_epochs = Entry(App, font=20, textvariable=epochs, width=16)
entry_epochs.grid(row=6, column=2)

# label and entry for mse
mse = DoubleVar()

label_mse = Label(App, text="MSE Threshold", font=25)
label_mse.grid(row=7, column=1)

entry_mse = Entry(App, font=20, textvariable=mse, width=16)
entry_mse.grid(row=7, column=2)

# label and Radiobutton for bias
bias = BooleanVar(value=False)

label_bias = Label(App, text="Bias", font=25)
label_bias.grid(row=8, column=1, padx=5, pady=5)

Radiobutton_bias = Radiobutton(App, text="YES", variable=bias, value=True)
Radiobutton_bias.grid(row=9, column=1, padx=2)

Radiobutton_bias = Radiobutton(App, text="NO", variable=bias, value=False)  # ,command=getValues())
Radiobutton_bias.grid(row=10, column=1, padx=2)

# Create a button to Train
button_Train = Button(App, text="Train", width=20, command=Train, font=15, pady=3)
button_Train.grid(row=11, column=1, padx=30)

# Create a button to visualize
button_Visualize = Button(App, text="visualize", width=20, command=visualize, font=15, pady=3)
button_Visualize.grid(row=11, column=2)

# Create a button to Test
button_Test = Button(App, text="Test", width=20, command=Test, font=15, pady=5)
button_Test.grid(row=14, column=1)


App.mainloop()