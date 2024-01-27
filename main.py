import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from Backpropagation import *
import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox


###############################################
# prepare data
def fill_na(dataset):
    # fill null values
    dataset['Area'].fillna(dataset['Area'].mean(), inplace=True)
    dataset['Perimeter'].fillna(dataset['Perimeter'].mean(), inplace=True)
    dataset['MajorAxisLength'].fillna(dataset['MajorAxisLength'].mean(), inplace=True)
    dataset['MinorAxisLength'].fillna(dataset['MinorAxisLength'].mean(), inplace=True)
    dataset['roundnes'].fillna(dataset['roundnes'].mean(), inplace=True)
    dataset[['BOMBAY', 'CALI', 'SIRA']] = pd.get_dummies(dataset['Class'])
    dataset.drop(columns='Class', inplace=True)

    # normalize data
    scaler = MinMaxScaler()
    dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

    return dataset_scaled


def split(x, y):
    X_train = pd.concat([x[0:30], x[50:80], x[100:130]], axis=0).reset_index(drop=True)
    y_train = pd.concat([y[0:30], y[50:80], y[100:130]], axis=0).reset_index(drop=True)
    X_test = pd.concat([x[30:50], x[80:100], x[130:150]], axis=0).reset_index(drop=True)
    y_test = pd.concat([y[30:50], y[80:100], y[130:150]], axis=0).reset_index(drop=True)
    return X_train, y_train, X_test, y_test


#######################################################################################################################
# Gui code tkinter
gui_data = []


def submit():
    if (not num_hidden_layers.get() or not number_of_neurons.get() or not learning_rate.get() or
            not number_of_epochs.get() or not activation_function.get()):
        tkinter.messagebox.showwarning(title="Error", message="Please enter a value for each field.")
    else:
        numb_hidden_retrieved = num_hidden_layers.get()
        number_of_neurons_retrieved = number_of_neurons.get()
        learning_rate_retrieved = learning_rate.get()
        epochs_retrieved = number_of_epochs.get()
        activation_function_retrieved = activation_function.get()
        model_bias = bias_retrieved.get()

        gui_data.append(str(numb_hidden_retrieved))
        gui_data.append(str(number_of_neurons_retrieved))
        gui_data.append(str(learning_rate_retrieved))
        gui_data.append(str(epochs_retrieved))
        gui_data.append(str(activation_function_retrieved))
        gui_data.append(str(model_bias))
        tkinter.messagebox.showinfo(title="Message",
                                    message="Submit Successfully")

        ##########################################################################
        # main logic code for back propagation
        # Loading Data
        dataset = pd.read_excel(
            "C:/Users/Malak/Documents/Semester 7/Neural Networks & Deep Learning/Tasks/Task 2/Dry_Bean_Dataset.xlsx",
            engine='openpyxl')
        dataset = fill_na(dataset)

        x = dataset[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']]
        y = dataset[['BOMBAY', 'CALI', 'SIRA']]

        X_train, y_train, X_test, y_test = split(x, y)

        data_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)

        # shuffle data_train
        data_train = data_train.sample(frac=1).reset_index(drop=True)
        X_train, y_train = (
            data_train[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']],
            data_train[['BOMBAY', 'CALI', 'SIRA']]
        )

        # Initialization
        num_features = 5
        num_classes = 3

        # Retrieve Data From GUI
        time.sleep(3)
        data = gui_data

        num_hidden_layers1 = int(data[0])
        number_of_neurons1 = data[1]
        neurons1 = number_of_neurons1.split(",")
        neurons1 = list(map(int, neurons1))
        learning_rate1 = float(data[2])
        epochs1 = int(data[3])
        activation_function1 = data[4]
        bias1 = data[5]

        print("Num OF Hidden Layers ==> ", num_hidden_layers1)
        print("Num Of Neurons ==> ", neurons1)
        print("Learning Rate ==> ", learning_rate1)
        print("Number Of Epochs ==> ", epochs1)
        print("Activation Function ==> ", activation_function1)
        print("Bias ==> ", bias1)

        weights, biases = train_algo(X_train.values, y_train.values, num_features, num_classes, num_hidden_layers1,
                                     neurons1, learning_rate1, epochs1, activation_function1, bias1)

        y_pred_test = predict(X_test.values, weights, biases, activation_function1)

        print("------------------------accuracy of test----------------------------------")
        predicted_labels = np.argmax(y_test.values, axis=1)
        Calc_accuracy(num_classes, predicted_labels, y_pred_test)

        print("------------------------Over-All accuracy ----------------------------------")
        y_pred_all = predict(x.values, weights, biases, activation_function1)
        predicted_labels = np.argmax(y.values, axis=1)
        Calc_accuracy(num_classes, predicted_labels, y_pred_all)

    return gui_data


################################################################################################################
# gui code
text_color = "#2f452b"
header_font = "18"
text_font = "6"

# Create Form
form = Tk()
form.geometry("600x500")
form.title("Task 2")

# Create Frame 3 Parameters
###########################
frame = Frame(form)
frame.pack()
model_inputs = LabelFrame(frame, text="Inputs", font=header_font, fg=text_color)
model_inputs.grid(row=0, column=0, padx=10, pady=10)


lb1 = Label(model_inputs, text="Number of Hidden Layers ", font=text_font, fg=text_color)
lb1.grid(row=0, column=0)
num_hidden_layers = Entry(model_inputs, width="29")
num_hidden_layers.grid(row=0, column=1)


lb2 = Label(model_inputs, text="Number of Neurons ", font=text_font, fg=text_color)
lb2.grid(row=1, column=0)
number_of_neurons = Entry(model_inputs, width="29")
number_of_neurons.grid(row=1, column=1)


lb3 = Label(model_inputs, text="Learning Rate  ", font=text_font, fg=text_color)
lb3.grid(row=2, column=0)
learning_rate = Entry(model_inputs, width="29")
learning_rate.grid(row=2, column=1)


lb4 = Label(model_inputs, text="Epochs ", font=text_font, fg=text_color)
lb4.grid(row=3, column=0)
number_of_epochs = Entry(model_inputs, width="29")
number_of_epochs.grid(row=3, column=1)


lb5 = Label(model_inputs, text="Activation Function", font=text_font, fg=text_color)
lb5.grid(row=4, column=0)
functions = ["Sigmoid", "Hyperbolic Tangent"]
activation_function = ttk.Combobox(model_inputs, values=functions, width="30")
activation_function.grid(row=4, column=1)


bias_retrieved = tkinter.StringVar(value="False")
bias = Checkbutton(model_inputs, text="Add bias", variable=bias_retrieved, onvalue="True", offvalue="False",
                   font=text_font,
                   fg=text_color)
bias.grid(row=5, column=0)


for widget in model_inputs.winfo_children():
    widget.grid_configure(padx=35, pady=10)


btn1 = Button(frame, command=submit, text="Submit", width="25", height="1", font=text_font, bg="#b5ceb1",
              fg=text_color)
btn1.grid(row=6, column=0, padx=20, pady=10)


# to show this form
form.mainloop()

