# Deep-Learning-algorithms
Implementation of all basic algorithms needed in  Deep Learning 

1) ## Logistic_regression_using_NN.ipynb

    * It is a simple logistic regression algorithm developed using NN (Neural Networks) with zero hidden layers
    * In this notebook, binary classification is done on the dataset of cats(cat or not cat) 
    
2) ## NN_with_1_hidden_layer.py
    * It's a python script file which contains all functions required to develop NN (Neural Networks) with one hidden layer
    * function designed are:
        * sigmoid(z)
        * initialize(n_x,n_h1,n_y)
        * forword_propagation(X,parameters)
        * evaluate_cost(A2,Y, parameters, lambd)
        * backword_propagation(X,Y,cache,parameters,lambd)
        * update_parameters(parameters,grads,learning_rate)
        * predict(parameters, X)
        * model(X_train, Y_train, X_test, Y_test,n_h1, num_iterations, learning_rate,lambd)
        * plot_cost(costs)
        
3) ## deep_NN_with_L_layers.py
    * It's a python script file which contains all functions required to develop NN (Neural networks) with 'n' hidden layer
    * It's the generalized algorithm for CNN
    * Functions designed are:
        * sigmoid(Z)
        * relu(Z)
        * sigmoid_backward(dA, cache)
        * relu_backward(dA, cache)
        * initialize_parameters(layer_dims)
        * linear_forward(A, W, b)
        * linear_activation_forward(A_prev, W, b, activation)
        * L_model_forward(X, parameters)
        * compute_cost(AL, Y)
        * linear_backward(dZ, cache)
        * linear_activation
        * linear_activation_backward(dA, cache, activation)
        * L_model_backward(AL, Y, caches)
        * update_parameters(parameters, grads, learning_rate)
        * predict(X, parameters)
        * L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost)
        * plot_cost(costs)
        
        
