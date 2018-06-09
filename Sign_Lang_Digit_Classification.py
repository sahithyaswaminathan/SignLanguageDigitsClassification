
# coding: utf-8

# In[2]:


import numpy as np
import os
import glob
from PIL import Image
import collections
import tensorflow as tf
from matplotlib import pyplot as py
import h5py
from scipy import ndimage, misc
import math


# In[3]:


folder = 'path'
files = glob.glob(folder)

images = []
img_arr = []
for names in files:
    img = Image.open(names)
    #img_arr[names] = np.asarray(img)
    img.convert('LA')
    image_resized = misc.imresize(img, (100, 100))
    images.append(image_resized)
    
images = np.stack(images, axis = 0)
print(images.shape)


# In[4]:


X_train = images.astype(float)
#print(Xtrain)
N = len(X_train)
print(N)


# In[5]:


#counting the number of classes ie number of files in each folder
        
n = 0  
for r, d, files in os.walk(path):  
    n+= len(files)  
    #print(files)  

print("count = {}".format(n))


# In[6]:


#assigning Ytrain with the values
ytrain = np.zeros(N)
ytrain[0:205] = 0
ytrain[205:411] = 1
ytrain[411:617] = 2
ytrain[617:823] = 3
ytrain[823:1030] = 4
ytrain[1030:1237] = 5
ytrain[1237:1444] = 6
ytrain[1444:1650] = 7
ytrain[1650:1858] = 8
ytrain[1858:2062] = 9

collections.Counter(ytrain)
print(ytrain.dtype)
ytrain = ytrain.astype(int)
print(ytrain.dtype)


# In[7]:


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# In[8]:


#Splitting the Train and test set
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, y_train, y_test = train_test_split(X_train, ytrain, test_size = 0.2, random_state = 0)

#normalizing the input vector
Xtrain = Xtrain/255
Xtest = Xtest/255
y_train = convert_to_one_hot(y_train, 10).T
print(y_train)
y_test = convert_to_one_hot(y_test,10).T
print ("number of training examples = " + str(Xtrain.shape[0]))
print ("number of test examples = " + str(Xtest.shape[0]))
print ("X_train shape: " + str(Xtrain.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(Xtest.shape))
print ("Y_test shape: " + str(y_test.shape))
print(y_train.shape[0])


# In[9]:


#Creating Placeholder
#The number of training examples are not defined at the moment
#Xtrain is defined with the dimension (None, n_H, n_W, n_C)
def createplaceholder(n_H0,n_W0,n_C0,n_y):
    X = tf.placeholder(tf.float32, shape = (None,n_H0,n_W0,n_C0))
    Y = tf.placeholder(tf.float32, shape = (None, n_y))
    
    return X, Y   


# In[10]:


#Xavier Initialization of weights W dimension = f x f x num of channels in previous layer x number of filters used
def initialize_parameters():
    W1 = tf.get_variable("W1",[5,5,3,32], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [3,3,32,50], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {"W1" : W1, "W2" : W2}
    print(W1, W2)
    return parameters


# In[11]:


tf.reset_default_graph() #Important so that we can reuse the W1/W2 term here
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 :" +str(parameters["W1"].eval()[1,1,1]))
    print(parameters["W1"].shape)


# In[12]:


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding = 'VALID')
    A1 = tf.nn.relu(Z1)
    M1 = tf.nn.max_pool(A1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID' )
    Z2 = tf.nn.conv2d(M1, W2, strides=[1,1,1,1], padding = 'VALID')
    A2 = tf.nn.relu(Z2)
    M2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID' )
    M2 = tf.contrib.layers.flatten(M2)
    Z3 = tf.contrib.layers.fully_connected(M2, 10, activation_fn = None) #to avert the default tf.nn.relu and softmax can be computed from cost function
    return Z3
    


# In[13]:


def compute_cost(Z3, Y):
    #yy = tf.transpose(y_train)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = tf.squeeze(Y))
    cost = tf.reduce_mean(cost)
    return cost


# In[14]:


def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]
    print(m)# number of training examples
    mini_batches = []
    np.random.seed(seed)
    print(Y.shape)
    print(Y)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[15]:


from tensorflow.python.framework import ops
def model(Xtrain, y_train, Xtest, y_test, learning_rate = 0.001, num_epochs = 100, minibatch_size = 50, print_cost = True):
    ops.reset_default_graph()
    (m, n_H0, n_W0, n_C0) = Xtrain.shape
    n_y = y_train.shape[1] #number of classes
    costs = []
    X, Y = createplaceholder(100,100,3,10)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    #initialize all the variables globally
    init = tf.global_variables_initializer()
    #Backpropogation
    with tf.Session() as sess:
        sess.run(init)
        for epochs in range(num_epochs):
            minibatch_cost = 0
            num_minibatch = int(m/minibatch_size)
            minibatchs = random_mini_batches(Xtrain, y_train, minibatch_size, seed = 0)
            for minibatch in minibatchs:
                (minibatch_X, minibatch_Y) = minibatch #selecting one batch
                print(minibatch_X.shape)
                _ , temp_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                minibatch_cost = minibatch_cost + (temp_cost/num_minibatch)
            if print_cost == True:
                print("Cost after epoch %i : %f" % (epochs, minibatch_cost))
                costs.append(minibatch_cost)
        #Calculating the prediction
        print("====Predicting the output===")
        predict = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict, tf.argmax(Y, 1))
        #Calculating the accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy :", accuracy)
        train_accuracy = accuracy.eval({X: Xtrain, Y: y_train})
        test_accuracy = accuracy.eval({X: Xtest, Y: y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters     


# In[16]:


_, _, parameters = model(Xtrain, y_train, Xtest, y_test)

