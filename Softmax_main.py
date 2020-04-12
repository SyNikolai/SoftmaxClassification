import sys
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns  
from data_generator import features, labels

sns.set_style('darkgrid')
np.set_printoptions(threshold= sys.maxsize) 

# Part A: Methods' definitions
def sigmoid(feature):
    """ Define the sigmoid function """
    return 1./(1. + np.exp(-feature))

def softmax(feature):
    """ Softmax function """
    return np.exp(feature) / np.sum(np.exp(feature), axis = 1, keepdims = True)

def hidden_layer(feature, weight, bh):
    """ Hidden layer function """
    return sigmoid(( feature @ weight) + bh)

def output_layer(hidden, w0, b0):
    """ Output layer function """
    return softmax(( hidden @ w0) + b0)

def nn(feature, weight, bh, w0, b0): 
    """ Define the neural network function """
    return output_layer(hidden_layer(feature, weight, bh), w0, b0)

def prediction(feature, weight, bh, w0, b0):            
    """ Define the neural network prediction function (results: 1 for class1, 0 for class2 ) """
    return np.around(nn(feature, weight, bh, w0, b0))

def loss(output, target):
    """ Loss function of the softmax layer ξ(X,T) = -SUM__N->number_of_samples(SUM__C->classes(t__nc*log(x_nc))) """
    return - (target * np.log(output)).sum()

def output_error(output, target):
    """ Error of the softmax function at the output """
    return output - target

def output_gradient_weight(hidden, out_error):
    """ Gradients for the weight parameters at the output layer """
    return  hidden.T @ out_error

def output_gradient_bias(out_error):
    """ Gradients for the output bias """
    return  np.sum(out_error, axis=0, keepdims = True)

def hidden_error(hidden, w0, out_error):
    """ Error of the hidden layer -> hidden * (1 - hidden) * (out_error . w0^T) """
    return np.multiply(np.multiply(hidden, (1 - hidden)), (out_error @ w0.T))

def hidden_gradient_weight(feature, hidden_error):
    """ Gradient for the weight parameters at the hidden layer """
    return feature.T @  hidden_error

def hidden_gradient_bias(hidden_error):
    """ Gradient for the bias parameters at the hidden layer """
    return  np.sum(hidden_error, axis = 0, keepdims = True)

class topology():
    """ Constructor for layer_i to layer_j connections """

    def __init__(self,  number_of_neurons_in, number_of_neurons_out, init_parameter):
        self.weights = np.random.randn(number_of_neurons_in, number_of_neurons_out) * init_parameter
        self.bias    = np.random.randn(1, number_of_neurons_out) * init_parameter

    def update(self, new_weights, new_bias):
        self.weights = new_weights
        self.bias    = new_bias

def backpropagation(x, target, hidden_weight, hidden_bias, output_weight, output_bias):
    """ Back propagation function """
    H                   = hidden_layer(x, hidden_weight, hidden_bias)           # Update Hidden layer parameters
    O                   = output_layer(H, output_weight, output_bias)           # Update Output layer parameters
    oe                  = output_error(O, target)                               # Update the output error
    Jacobi_out_weights  = output_gradient_weight(H, oe)                         # Calculate the Jacobi matrix of out_weights
    Jacobi_out_bias     = output_gradient_bias(oe)                              # Calculate the Jacobi matrix of out_bias               
    he                  = hidden_error(H, output_weight, oe)                    # Calculate the hidden layer's error
    Jacobi_hid_weights  = hidden_gradient_weight(x, he)                         # Calculate the Jacobi matrix of hid_weights
    Jacobi_hid_bias     = hidden_gradient_bias(he)                              # Calculate the Jacobi matrix of hid_bias
    
    return [Jacobi_hid_weights, Jacobi_hid_bias, Jacobi_out_weights, Jacobi_out_bias]

def momentum_calc(x, target, parameters, moments, momentum, learning_rate):
    """ Momentum calculation """
    # parameters = [hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias]
    Jacobis = backpropagation(x, target, *parameters)
    return [momentum * m - learning_rate * j for m, j in zip(moments, Jacobis)]

def new_parameters(parameters, moments):
    """ Update the parameters """
    return [a + b for a,b in zip(parameters, moments)]

# Part B: Create and Initialize the Neural Net and its parameters
if __name__ == '__main__':
    number_of_input_neurons             = 2
    number_of_hidden_neurons            = 3
    number_of_output_neurons            = 2
    init_parameter                      = 1
    layer12                             = topology(number_of_input_neurons,  number_of_hidden_neurons, init_parameter)                                               
    layer23                             = topology(number_of_hidden_neurons, number_of_output_neurons, init_parameter)                                               
    learning_rate                       = 0.0001                                                          # set the learning rate of the neural net
    momentum                            = 0.9                                                             # set the momentum value
    parameters                          = [layer12.weights, layer12.bias, layer23.weights, layer23.bias]  # define the list of parameters
    Moments                             = [np.zeros_like(m) for m in parameters]                                     

    # Gradient descent
    steps               = 1000                                                         # number of gradient descent updates
    # Save loss in each step 
    loss_list           = [loss(nn(features, *parameters), labels)]       
    for i in range(steps):
        Moments         = momentum_calc(features, labels, [layer12.weights, layer12.bias, layer23.weights, layer23.bias], Moments, momentum, learning_rate)  
        Wh, bh, Wo, bo  = new_parameters([layer12.weights, layer12.bias, layer23.weights, layer23.bias], Moments)                        
        layer12.update(Wh,bh)
        layer23.update(Wo,bo)
        loss_list.append(loss(nn(features, layer12.weights, layer12.bias, layer23.weights, layer23.bias), labels))

    # Plot the loss over the iterations
    fig = plt.figure(figsize = (6, 4))
    plt.plot(loss_list, 'b-')
    plt.xlabel('Iteration')
    plt.ylabel('$Loss$', fontsize = 12)
    plt.title('Loss over each iteration')
    plt.xlim(0, 300)
    fig.subplots_adjust(bottom = 0.2)
    fig = plt.gcf()
    fig.canvas.set_window_title('Gradient descent')
    plt.show()

    # Plot the resulting decision boundary
    grid_points = 400
    grid_x = np.linspace(-10, 10, num = grid_points)
    grid_y = np.linspace(-10, 10, num = grid_points)
    px, py = np.meshgrid(grid_x, grid_y) 
    # Initialize and fill the classification plane
    class_plane = np.zeros((grid_points, grid_points))
    for i in range(grid_points):
        for j in range(grid_points):
            Predicted_class = prediction(np.asarray([px[i,j], py[i,j]]),
             layer12.weights, layer12.bias, layer23.weights, layer23.bias)
            class_plane[i,j] =  Predicted_class[0, 0]


    # Create a color map to show the boundary
    plt.figure(figsize=(6, 4))
    plt.contourf(px, py, class_plane, cmap = 'plasma')
    plt.scatter(features[:,0], features[:,1], marker = '.', c = labels[:,0], cmap = 'binary', s = 70, alpha = 0.6) 
    plt.legend(loc=1)
    plt.xlabel('x feature', fontsize = 12)
    plt.ylabel('y feature', fontsize = 12)
    plt.axis([-5, 5, -5, 5])
    plt.title('Classification plane for the 2 types of data')
    fig = plt.gcf()
    fig.canvas.set_window_title('Classified Data')
    plt.show()