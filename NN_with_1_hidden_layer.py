from PIL import Image
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from numpy import *

'''
step 1: make list of elements of file

list_img = os.listdir(file_addr)

step 2: resize all images from input folder and store in input_resize folder

for i in list_img:
    im = Image.open(input_file_addr+'\\'+i)
    im1 = im.resize((200,200))
    im2 = im1.convert('RGB')
    im2.save(input_resize_file_addr+'\\'+i,'JPEG')   
    
step 3: flatten images to matrix(m,n_x)

X = np.array([np.array(Image.open(input_file_addr+'\\'+i)).flatten() for i in list_img],'f')
X = X.T

step 4: labelling the dataset

m = X.shape[1]           #no. of images
m_t = X_t.shape[1]
Y = np.zeros((m,1),dtype=int)
Y_t = np.zeros((m_t,1),dtype=int)
Y[0:m] = 1 

step 5: reshape Y to maintain the consistency

Y = Y.reshape((1,X.shape[1])).T

step 5: shuffle data (need to do it for better result)

X_train,Y_train = shuffle(X,Y, random_state=0)

step 6: standardize the data (not neccessary but good practice)

X_train = X_train/255   #255 is maximum possible value in image pixle 

step 7: do all above steps for test data

X_test = ....
Y_test = ....

step 8: run the model function

n_h1 = 7      #number of nodes in layer
d = model(X_train, Y_train, X_test, Y_test,n_h1, num_iterations = 6000, learning_rate = 0.05,lambd = 0)

step 9: Print train/test Errors
Y_prediction_train = d["Y_prediction_train"]
Y_prediction_test = d["Y_prediction_test"]

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

step 10: draw cost vs iteration graph

'''


#sigmoid function
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
   
    
#initialization of parameter w,b
def initialize(n_x,n_h1,n_y):
    W1 = np.random.randn(n_h1,n_x)*0.01
    b1 = np.zeros(shape=(n_h1,1))
       
    W2 = np.random.randn(n_y,n_h1)*0.01
    b2 = np.zeros(shape=(n_y,1))
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}
    return parameters


#forword and backword propagation
def forword_propagation(X,parameters):
    m = X.shape[1]
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    #forword propagation
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {'Z1':Z1,
             'A1':A1,
             'Z2':Z2,
             'A2':A2}
    
    return A2,cache


def evaluate_cost(A2,Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    #with reegulrization
    
    cost = (-1/m)* np.sum(Y * np.log1p(A2) + (1-Y) * (np.log1p(1-A2))) + (lambd/(2*m))*(np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return cost


def backword_propagation(X,Y,cache,parameters,lambd):
    A2 = cache['A2']
    A1 = cache['A1']
    
    W2 = parameters['W2']
    W1 = parameters['W1']
    
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T) + (1/m)*(lambd * W2)     #with reegulrization
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    
    dZ1 = np.dot(W2.T,dZ2)*(1-np.square(A1))
    dW1 = (1/m)*np.dot(dZ1,X.T) + (1/m)*(lambd * W1)     #with reegulrization
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    
    grads = {'dW1':dW1,
             'db1':db1,
             'dW2':dW2,
             'db2':db2}
    
    return grads

def update_parameters(parameters,grads,learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}
    
    return parameters

def predict(parameters, X):
    m= X.shape[1]
    A2,cache = forword_propagation(X, parameters)
    Y_prediction = np.zeros(shape=(1,m))
    for i in range(A2.shape[1]):
        Y_prediction[0, i] = 1 if A2[0, i] > 0.5 else 0
       
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test,n_h1, num_iterations=2000, learning_rate=0.5,lambd = 0.7):
    
    # initialize parameters
    parameters = initialize(X_train.shape[0],n_h1,1)
    costs = []

    for i in range(num_iterations):
       # learning_rate = learning_rate/(1+(i*0.05))    #learning rate decay
        
        A2,cache = forword_propagation(X_train,parameters)
        
        cost = evaluate_cost(A2,Y_train, parameters, lambd)
        
        grads = backword_propagation(X_train,Y_train,cache,parameters, lambd)
        
        parameters = update_parameters(parameters,grads,learning_rate)
        
        #if i%100==0:
        costs.append(cost)
        print ("Cost after iteration %i: %f" % (i, cost))

    # Predict test/train set examples
    Y_prediction_test = predict(parameters, X_test)
    Y_prediction_train = predict(parameters, X_train)
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def plot_cost(costs):
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
