import sys
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns  

sns.set_style('darkgrid')
np.random.seed(12)
np.set_printoptions(threshold= sys.maxsize)


#Random data generator (data clustered in specified space)
class random_data():
    """ Generates random data features (random x, random y) and its labels (label_x, label_y) """

    def __init__(self, observations, mu, sigma):

        self.features                = []                           # list of lists to save the observations (random data)
        self.mu                      = mu                           # center data around mu
        self.observations            = observations                 # number of observations (number of rows)
        self.sigma                   = sigma                        # standard deviation 

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
features  = np.delete(features, 2, 1)                       # comment: this part is messy, it will be revised in the future

def plot_data():
    plt.figure(figsize = (8, 6))
    plt.title('Data Visualization')    
    plt.scatter(features[:,0], features[:,1], marker = 'o', c = labels[:,0], cmap = 'plasma', s = 70, alpha = 0.6)  
    plt.xlabel('x feature', fontsize = 12)
    plt.ylabel('y feature', fontsize = 12)
    fig = plt.gcf()
    fig.canvas.set_window_title('Data generator')
    plt.show()
if __name__ == '__main__':
 plot_data()