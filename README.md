# SoftmaxClassification
A neural network that classifies random 2-dimensional data with a softmax output classifier. 

 - [Introduction]

    What does softmax mean?
    
    The softmax function, also known as softargmax or normalized exponential function,converts a vector of K real numbers into a probability distribution of K possible outcomes. It is a generalization of the logistic function to multiple dimensions, and used in multinomial logistic regression. The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes, based on Luce's choice axiom.

    In this project, a neural network was built from scratch following a traditional algorithmic recipe, using a softmax function
    as the activation function.

 - [Data Generator]
    
    For the purposes of the project, a dummy data class was created, which can be found in the data_generator.py script. Running that script will result in a distribution of data of two different categories (labeled). The distribution can be easily customized on the script for futher exploration.
        
 - [Softmax Script]
    
    The Softmax_main.py script contains all the necessery fucntion for building a neural network, such as activation function, loss calculation, gradient descent calculation, momentum calculation, a topology of the neural network and plotting functions.

    Simply running the script should produce a loss graph and a decision boundary for the data generated previously.

    


