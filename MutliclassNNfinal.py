import sys
import numpy as np  
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns  

sns.set_style('darkgrid')
np.random.seed(33)
np.set_printoptions(threshold= sys.maxsize) 

# Part A: Random data creation
class random_data():
    """ Generates random data features (random x, random y) in the format of a list of lists """

    def __init__(self, observations, mu, sigma):

        self.features                = []                           # list of lists to save the observations (random data)
        self.mu                      = mu                           # center data around mu
        self.observations            = observations                 # number of observations (number of rows)
        self.sigma                   = sigma                        # standard deviation ("noise of the data")

        for i in range(self.observations):
                    self.x_feature = self.random()
                    self.y_feature = self.random()
                    self.labeling_method()
                    self.features.append([self.x_feature, self.y_feature , self.label_x, self.label_y])   # this part is hard coded for 2D
                    
    def random(self):
        return self.sigma * np.random.randn()   +   self.mu 

    def labeling_method(self):        
        """ method to assign a class to the generated data """
        """ euclidean distance is considered as the criteria here """
        if np.sqrt(self.x_feature*self.x_feature + self.y_feature*self.y_feature) < 0.3 :
            self.label_x = 1
            self.label_y = 0
        else:
            self.label_x = 0
            self.label_y = 1
            
D1 = random_data(500,  0,   0.15)        # Create 2 sets of data, each set represents a different class of data
D2 = random_data(500,  0,   0.11)
features  = np.vstack(D1.features + D2.features)                 #make this part better 
labels_x  = features[:, 2]
labels_y  = features[:,3]
labels    = np.column_stack((labels_x,labels_y))
features  = np.delete(features, 3, 1)
features  = np.delete(features, 2, 1)

# Plot the data
plt.figure(figsize = (8, 6))  
plt.scatter(features[:,0], features[:,1], marker = 'o', c = labels[:,0], cmap = 'plasma', s = 70, alpha = 0.6)  
plt.xlabel('x feature', fontsize = 12)
plt.ylabel('y feature', fontsize = 12)
plt.show()

# Part B: Methods' definitions
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
    """ Gradient for the bias parameters at the output layer """
    return  np.sum(hidden_error, axis = 0, keepdims = True)

class topology():
    """ constructor for layer_i to layer_j connections """

    def __init__(self,  number_of_neurons_in, number_of_neurons_out, init_parameter):
        self.weights = np.random.randn(number_of_neurons_in, number_of_neurons_out) * init_parameter
        self.bias    = np.random.randn(1, number_of_neurons_out) * init_parameter

    def update(self, new_weights, new_bias):
        self.weights = new_weights
        self.bias    = new_bias

def backpropogation(x, target, hidden_weight, hidden_bias, output_weight, output_bias):
    """ Back propogation function """
    H                   = hidden_layer(x, hidden_weight, hidden_bias)           # Update Hidden layer parameters
    O                   = output_layer(H, output_weight, output_bias)           # Update Output layer parameters
    oe                  = output_error(O, target)                               # Update the output error
    Jacobi_out_weights  = output_gradient_weight(H, oe)                         # Calculate the Jacobi matrix of out_weights
    Jacobi_out_bias     = output_gradient_bias(oe)                              # Calculate the Jacobi matrix of out_bias               
    he                  = hidden_error(H, output_weight, oe)
    Jacobi_hid_weights  = hidden_gradient_weight(x, he)
    Jacobi_hid_bias     = hidden_gradient_bias(he)
    
    return [Jacobi_hid_weights, Jacobi_hid_bias, Jacobi_out_weights, Jacobi_out_bias]

def momentum_calc(x, target, parameters, moments, momentum, learning_rate):
    """ Momentum calculation """
    # parameters = [hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias]
    Jacobis = backpropogation(x, target, *parameters)
    return [momentum * m - learning_rate * j for m, j in zip(moments, Jacobis)]

def new_parameters(parameters, moments):
    """ Update the parameters """
    return [a + b for a,b in zip(parameters, moments)]


# Part C: Create and Initialize the Neural Net and its parameters
layer12       = topology(2,3,1)                                                 # input layer connections to hidden layer1
layer23       = topology(3,2,1)                                                 # hidden layer1 connections to output layer
learning_rate = 0.01                                                            # set the learning rate of the neural net
momentum      = 0.9                                                             # set the momentum value
parameters    = [layer12.weights, layer12.bias, layer23.weights, layer23.bias]  # define the list of parameters
Moments       = [np.zeros_like(m) for m in parameters]                                     
# Moments = [Moments Hidden weights, Moments Hidden bias, Moments Output weights, Moments Output bias]


# Gradient descent
steps               = 300                                                         # number of gradient descent updates
#new_learning_rate   = learning_rate / steps                                       # update learning rate 


### DEBUGGING MY ERROR IS INCREASING -> FIX IT  ###
# Save loss in each step 
loss_list           = [loss(nn(features, *parameters), labels)]         # WORKS FINE

for i in range(steps):
    Moments         = momentum_calc(features, labels, parameters, Moments, momentum, learning_rate)  
    Wh, bh, Wo, bo  = new_parameters(parameters, Moments)                         # HERE!!change the variables into the structures
    layer12.update(Wh,bh)
    layer23.update(Wo,bo)
    loss_list.append(loss(nn(features, layer12.weights, layer12.bias, layer23.weights, layer23.bias), labels))



# Plot the loss over the iterations
fig = plt.figure(figsize=(5, 3))
plt.plot(loss_list, 'b-')
plt.xlabel('iteration')
plt.ylabel('$\\xi$', fontsize=12)
plt.title('Decrease of loss over backprop iteration')
plt.xlim(0, 300)
fig.subplots_adjust(bottom=0.2)
plt.show()


# Plot the resulting decision boundary
# Generate a grid over the input space to plot the color of the
#  classification at that grid point
nb_of_xs = 200
xs1 = np.linspace(-2, 2, num=nb_of_xs)
xs2 = np.linspace(-2, 2, num=nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2) # create the grid
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        pred = prediction(
            np.asarray([xx[i,j], yy[i,j]]), layer12.weights, layer12.bias, layer23.weights, layer23.bias)
        classification_plane[i,j] = pred[0, 0]
# Create a color map to show the classification colors of each grid point
plt.figure(figsize=(6, 4))
# Plot the classification plane with decision boundary and input samples
plt.contourf(xx, yy, classification_plane, cmap= 'plasma')
# Plot both classes on the x1, x2 plane
plt.scatter(features[:,0], features[:,1], marker = 'o', c = labels[:,0], cmap = 'plasma', s = 70, alpha = 0.6) 
plt.legend(loc=1)
plt.xlabel('x feature', fontsize = 12)
plt.ylabel('y feature', fontsize = 12)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.title('red star vs blue circle classification boundary')
plt.show()



