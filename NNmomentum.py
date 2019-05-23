import sys
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns  

sns.set_style('darkgrid')
np.random.seed(12)
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
        """ distance from axis start is considered as the criteria here """
        if self.x_feature + self.y_feature > 0.6 or self.x_feature + self.y_feature < 0.2:
            self.label_x = 1
            self.label_y = 0
        else:
            self.label_x = 0
            self.label_y = 1
            
D1 = random_data(500,  0,   0.3)                            # Create 2 sets of data, each set represents a different class of data
D2 = random_data(500,  2,   0.3)
features  = np.vstack(D1.features + D2.features)            # Stack the features 
labels    = np.column_stack((features[:,2],features[:,3]))  # Create a list containing only the labels for each element
features  = np.delete(features, 3, 1)                       # we dont need the labels in the features list anymore
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
    he                  = hidden_error(H, output_weight, oe)
    Jacobi_hid_weights  = hidden_gradient_weight(x, he)
    Jacobi_hid_bias     = hidden_gradient_bias(he)
    
    return [Jacobi_hid_weights, Jacobi_hid_bias, Jacobi_out_weights, Jacobi_out_bias]

def momentum_calc(x, target, parameters, moments, momentum, learning_rate):
    """ Momentum calculation """
    # parameters = [hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias]
    Jacobis = backpropagation(x, target, *parameters)
    return [momentum * m - learning_rate * j for m, j in zip(moments, Jacobis)]

def new_parameters(parameters, moments):
    """ Update the parameters """
    return [a + b for a,b in zip(parameters, moments)]


# Part C: Create and Initialize the Neural Net and its parameters
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
#new_learning_rate   = learning_rate / steps                                       # update learning rate 
# Save loss in each step 
loss_list           = [loss(nn(features, *parameters), labels)]       

for i in range(steps):
    Moments         = momentum_calc(features, labels, [layer12.weights, layer12.bias, layer23.weights, layer23.bias],
                                    Moments, momentum, learning_rate)  
    Wh, bh, Wo, bo  = new_parameters([layer12.weights, layer12.bias, layer23.weights, layer23.bias], Moments)                         # HERE!!change the variables into the structures
    layer12.update(Wh,bh)
    layer23.update(Wo,bo)
    loss_list.append(loss(nn(features, layer12.weights, layer12.bias, layer23.weights, layer23.bias), labels))


# Plot the loss over the iterations
fig = plt.figure(figsize = (6, 4))
plt.plot(loss_list, 'b-')
plt.xlabel('iteration')
plt.ylabel('$loss$', fontsize = 12)
plt.title('loss over each iteration')
plt.xlim(0, 300)
fig.subplots_adjust(bottom = 0.2)
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
plt.show()


