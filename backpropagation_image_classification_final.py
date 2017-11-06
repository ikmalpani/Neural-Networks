import os
import path
import struct
import math
import numpy as np
import matplotlib.pyplot as plt


# Source for reading the idx files as numpy arrays: https://gist.github.com/tylerneylon
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

# Data Source: http://yann.lecun.com/exdb/mnist/
train_data = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')
test_data = read_idx('t10k-images.idx3-ubyte')
test_labels = read_idx('t10k-labels.idx1-ubyte')

def convert_labels(data):
    new_data = []
    for i in range(len(data)):
        temp = [0]*10
        temp[data[i]] = 1
        new_data.append(temp)
    return new_data

train_labels = convert_labels(train_labels)
test_labels = convert_labels(test_labels)

# Based on Xavier Normal initialization 
w_input = np.random.uniform(low=-0, high=1, size=(784, 10))* np.sqrt(6/(784+10))
w_layer1 = np.random.uniform(low=-0, high=1, size=(10, 10))* np.sqrt(6/(10+10))
w_layer1_bias = np.random.uniform(low=-0, high=1, size=(10,1))* np.sqrt(6/(10+10))
w_layer2_bias = np.random.uniform(low=-0, high=1, size=(10,1))* np.sqrt(6/(10+10))

# feed-forward activation functions - hyperbolic tangent
def act_fun(v):
    return np.tanh(v)

# feedback activation function - hyperbolic tangent
def derv_act_fun(v):
    return (1 - np.tanh(v)**2)

def feedforward(input_data, bias, weight):
    local_ind_field = np.dot(weight.T,input_data) + bias
    output = act_fun(local_ind_field)
    return local_ind_field, output

eta = 4
training = []
testing = []
energy_training = []
energy_testing = []
while(True):
    train_correct = 0
    test_correct = 0
    layer1_local_field = []
    layer1_output = []
    layer2_local_field = []
    layer2_output = []
    for i in range(len(train_data)):
        xi = train_data[i]
        xi.resize(784, 1)
        xi = normalize(xi)
        local_ind_field, output = feedforward(xi, w_layer1_bias, w_input)
        layer1_local_field.append(local_ind_field)
        layer1_output.append(output)
        
        local_ind_field, output = feedforward(output, w_layer2_bias, w_layer1)
        layer2_local_field.append(local_ind_field)
        layer2_output.append(output)
        
        max_index = output.argmax(axis=0)[0]
        new_output = [0]*10
        new_output[max_index] = 1
        x = np.linalg.norm(np.asarray(train_labels[i]) - np.asarray(new_output))**2
        if x==0:
            train_correct += 1
        e = 2 * np.subtract(np.asarray(train_labels[i]).reshape(10,1), output)/len(train_data)
        
        local_ind_field = local_ind_field.reshape(10,)
        e = e.reshape(10,)
        w_layer2_bias_grad = - eta * np.asarray([e[i]*derv_act_fun(local_ind_field)[i] for i in range(10)]).reshape(10,1)

        w_layer1_grad = - eta * np.dot(layer1_output[i], np.asarray([e[i]*derv_act_fun(local_ind_field)[i] for i in range(10)]).reshape(1,10))
        
        w_layer1_bias_grad = -eta * np.dot(np.dot(layer1_output[i], np.asarray([e[i]*derv_act_fun(local_ind_field)[i] for i in range(10)]).reshape(1,10)), derv_act_fun(layer1_local_field[i]))
        w_input_grad = - eta * np.dot(xi , np.dot(np.dot(layer1_output[i], np.asarray([e[i]*derv_act_fun(local_ind_field)[i] for i in range(10)]).reshape(1,10)), derv_act_fun(layer1_local_field[i])).reshape(1,10))

        # update weights
        w_input = np.subtract(w_input, w_input_grad)
        w_layer1_bias = np.subtract(w_layer1_bias, w_layer1_bias_grad)
        w_layer1 = np.subtract(w_layer1, w_layer1_grad)
        w_layer2_bias = np.subtract(w_layer2_bias, w_layer2_bias_grad)

    training_accuracy = train_correct/len(train_data)
    training.append(len(train_data) - train_correct)
    mse = 0
    for i in range(len(train_data)):
        mse += np.linalg.norm(layer2_output[i] - train_labels[i])**2
    mse = mse/len(train_data)
    energy_training.append(mse)
    print ("Root mean square Error:",mse,"Training accuracy:", training_accuracy, "No. of misclassifications:", (len(train_data) - train_correct))
    for i in range(len(test_data)):
        xi = test_data[i]
        xi.resize(784, 1)
        xi = normalize(xi)
        local_ind_field, output = feedforward(xi, w_layer1_bias, w_input)
        layer1_local_field.append(local_ind_field)
        layer1_output.append(output)
        
        local_ind_field, output = feedforward(output, w_layer2_bias, w_layer1)
        layer2_local_field.append(local_ind_field)
        layer2_output.append(output)
        
        max_index = output.argmax(axis=0)[0]
        new_output = [0]*10
        new_output[max_index] = 1
        x = np.linalg.norm(np.asarray(test_labels[i]) - np.asarray(new_output))**2
        if x==0:
            test_correct += 1
    testing_accuracy = test_correct/len(test_data)
    testing.append(len(test_data) - test_correct)
    mse = 0
    for i in range(len(test_data)):
        mse += np.linalg.norm(layer2_output[i] - test_labels[i])**2
    mse = mse/len(test_data)
    energy_testing.append(mse)
    print ("Root mean square Error:",mse,"Testing accuracy:", testing_accuracy, "No. of misclassifications:", (len(test_data) - test_correct))
    if testing_accuracy>0.95:
        break

fig, ax = plt.subplots(figsize=(10,10))
ax.set_ylim([0,60000])
plt.xlabel('Number of Epochs')
plt.ylabel('Number of Misclassifications')
plt.plot(range(len(training)), training, c = 'green', label='Training misclassifications')
plt.plot(range(len(testing)), testing, c = 'blue', label='Testing misclassifications')
plt.legend(loc = 'best')
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
plt.xlabel('Number of Epochs')
plt.ylabel('Energies')
plt.plot(range(len(energy_training)), energy_training, c = 'green', label='Training energies')
plt.plot(range(len(energy_testing)), energy_testing, c = 'blue', label = 'Testing energies')
plt.legend(loc = 'best')
plt.show()
