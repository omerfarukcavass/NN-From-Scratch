
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import time
import pickle

# load mnist dataset using keras dataset loader
def load_dataset():
    # split the dataset between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range as usual
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Convert image shapes to (28, 28, 1). (add number of channel dimension)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    # one hot encoding for class labels using keras function
    num_classes=10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train,x_test,y_train,y_test

# relu activation forward pass
# Z could by any dimensional matrix
def relu_forward(Z):
    A = np.maximum(0,Z)
    cache = Z # cache Z to use in backward pass
    return A, cache

# softmax forward pass
# z: (10,m) matrix in this network
# Note: Z is not needed in cache since we will directly compute gradient
# with respect to Z not A for the output layer.
def softmax_forward(Z):
    exps=np.exp(Z)
    return exps/np.sum(exps,axis=0)  # axis=0 enables multiple samples

# relu backward pass
# Z is taken from forward pass to calculate derivative
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # positive side
    dZ[Z <= 0] = 0 # negative side
    return dZ

# backward for softmax + cross entropy
# finds directly dZ (derivative of loss with respect to last linear output)
# p: probability predictions, y: true classes
# p:(10,m), y:(10,m) in this model
def softmax_crossentropy_backward(p,y):
    # dL/dz = p-y
    dZ=p-y
    return dZ

# initialize weights and biases using normal dist.
def initialize_parameters():

    np.random.seed(1) # for reproducibility

    # Convolution layers
    f, f, n_C_prev, n_C = 5,5,1,4
    W1 = np.random.randn(f, f, n_C_prev, n_C)*0.1
    b1 = np.random.randn(1, 1, 1, n_C)*0.1

    f, f, n_C_prev, n_C = 5,5,4,8
    W2 = np.random.randn(f, f, n_C_prev, n_C)*0.1
    b2 = np.random.randn(1, 1, 1, n_C)*0.1

    # FC layers
    h,h_prev = 128,128
    W3 = np.random.randn(h,h_prev)*0.1
    b3 = np.random.randn(h, 1)*0.1

    h,h_prev = 10,128
    W4 = np.random.randn(h, h_prev)*0.1
    b4 = np.random.randn(h, 1)*0.1

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                 }

    print("Parameters initialized.")
    print("W1 shape:",parameters["W1"].shape)
    print("b1 shape:",parameters["b1"].shape)
    print("W2 shape:",parameters["W2"].shape)
    print("b2 shape:",parameters["b2"].shape)
    print("W3 shape:",parameters["W3"].shape)
    print("b3 shape:",parameters["b3"].shape)
    print("W4 shape:",parameters["W4"].shape)
    print("b4 shape:",parameters["b4"].shape)

    return parameters

# forward pass for the linear output
# cache A_prev, W and b to use in backward pass
def linear_forward(A_prev, W, b):
    Z = W.dot(A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

# complete forward pass for given activation (linear + activation)
# store A_prev, W, b and Z in cache for backward pass
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = softmax_forward(Z) # no cache needed
        activation_cache = None

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_forward(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

# calculate cross entropy loss function
#AL and Y are matrix of shape (10, m) in this network
def compute_crossentropy_cost(AL,Y):
    m = Y.shape[1]
    loss=-np.sum(Y*np.log(AL))  # element wise multiplication then sum
    return loss/float(m)

# backward pass for the linear part
# A_prev and W are taken from forward pass to find derivatives
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

# backward pass for linear + activation
# use caches from forward pass
def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache

    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# convert shape of the last pooling layer to use in FC layers
#  A of shape (m,4,4,8) becomes (128,m) in this network
def flatten_2D_output(A):
    A_new=np.moveaxis(A, 0, -1)  # (4,4,8,m)
    A_new=A_new.reshape((128,-1)) # (128,m)
    return A_new

# this is the inverse of flatten operation for  backward pass
# dA (128,m) -> (m,4,4,8) in this network
def deflatten_to_2D(dA):
    dA_new=np.moveaxis(dA, 0, -1) # (m,128)
    dA_new=dA_new.reshape((-1,4,4,8)) # (m,4,4,8)
    return dA_new

# single conv. operation: dot product of a slice with filter then adding bias
# a_slice: slice of input data of shape (f, f, n_C_prev)
# W: Weights of shape (f, f, n_C_prev)
# b: Bias of shape (1, 1, 1) -- one scalar for a filter

def conv_operation(a_slice, W, b):
    # dot product
    s = np.multiply(a_slice,W)
    Z = np.sum(s)

    # Add bias b
    b = np.squeeze(b)
    Z = Z + b
    return Z

# forward pass for convolution layer (linear part)
# A_prev: output of prev. layer (m, n_H_prev, n_W_prev, n_C_prev)
# W: Weights (f, f, n_C_prev, n_C)
# b: Biases (1, 1, 1, n_C)

def conv_forward(A_prev, W, b):
    # find dims.
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    # Find output shape
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1

    # output matrix
    Z = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):   # for each training example
        a_prev = A_prev[i]
        for c in range(n_C):   # for each filter
            for h in range(n_H):
                vert_start = h
                vert_end = vert_start  + f
                for w in range(n_W):
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # a slice from previous output matrix
                    a_slice_prev = a_prev[vert_start:vert_end,horiz_start:horiz_end,:]

                    # apply convolution operation
                    weights = W[:, :, :, c]
                    biases  = b[:, :, :, c]
                    Z[i, h, w, c] = conv_operation(a_slice_prev, weights, biases)

    # cache for backward pass
    cache = (A_prev, W, b)
    return Z, cache

# complete forward pass for a conv. layer (linear + activation)
def conv_activation_forward(A_prev, W, b):
    Z, linear_cache = conv_forward(A_prev, W, b)
    A, activation_cache = relu_forward(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

# max pooling operation
# A_prev: output of prev. layer (m, n_H_prev, n_W_prev, n_C_prev)
# Output shape: (m, n_H, n_W, n_C_prev)
def pool_forward(A_prev):
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # set filter size and stride
    f,stride = 2,2

    # set dims of the output
    n_H = int((n_H_prev - f) / stride +1)
    n_W = int((n_W_prev - f) / stride +1)
    n_C = n_C_prev

    # output matrix
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):   # for each training examples
        a_prev_slice = A_prev[i]
        for c in range (n_C):  # for each channel
            for h in range(n_H):
                vert_start = stride * h
                vert_end = vert_start + f
                for w in range(n_W):
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    # a slice from previous output
                    a_slice_prev = a_prev_slice[vert_start:vert_end,horiz_start:horiz_end,c]
                    A[i, h, w, c] = np.max(a_slice_prev)
    # cache for backward pass
    cache = A_prev
    return A, cache

# backward pass for the convolution layer
# dZ : derivative of loss wrt linear output
# cache : A_prev, W, b from forward pass
def conv_backward(dZ, cache):

    # take A_prev, W, b
    (A_prev, W, b) = cache
    # find dims
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape

    # initialize dA_prev,dW,db
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(m): # for each training example
        a_prev = A_prev[i]
        da_prev = dA_prev[i]
        for c in range(n_C):
            for h in range(n_H):
                vert_start = h
                vert_end = vert_start + f
                for w in range(n_W):
                    horiz_start = w
                    horiz_end = horiz_start + f
                    # a slice from prev. output
                    a_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,:]

                    # find gradients
                    da_prev[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c] # sum over channels
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]

    # Average over training examples
    dW=(1./m) * dW
    db=(1./m) * db
    return dA_prev, dW, db

# complete backward pass for conv. layer
# relu backward used since both conv.layers are relu activated.
def conv_activation_backward(dA,cache):
    linear_cache, activation_cache = cache
    dZ = relu_backward(dA, activation_cache)
    dA_prev,dW,db = conv_backward(dZ, linear_cache)
    return dA_prev,dW,db

# backward pass for max pooling
# dA_prev = dA for the max item in slice, 0 otw
def pool_backward(dA, cache):

    # take A_prev to find max element
    A_prev = cache
    stride,f = 2,2

    #find dims
    m, n_H_prev, n_W_prev, n_C_prev =A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # initialize dA_prev
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m): # for each training example
        a_prev = A_prev[i,:,:,:]
        for c in range(n_C):
            for h in range(n_H):
                vert_start  = h*stride
                vert_end    = h*stride+f
                for w in range(n_W):
                    horiz_start = w*stride
                    horiz_end   = w*stride+f
                    # a slice from previous layer output
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c ]

                    #  a 1-0 mask. 1 for max item, 0 otw
                    mask = a_prev_slice == np.max(a_prev_slice)
                    # find derivatinve using mask
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
    return dA_prev

# update parameters (weights, biases)
# using SGD without momentum
def update_parameters(parameters, grads, learning_rate):
    for i in range(4):
        parameters["W"+str(i+1)] = parameters["W"+str(i+1)]-learning_rate*grads["dW"+str(i+1)]
        parameters["b"+str(i+1)] = parameters["b"+str(i+1)]-learning_rate*grads["db"+str(i+1)]
    return parameters

# create mini batches for SGD
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    # reshape X and y into tabular format
    # (to horizontally stack)
    X=X.reshape((X.shape[0],-1))
    y=y.reshape((y.shape[0],-1))

    # stack horizontally to shuffle
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-10]
        X_mini = X_mini.reshape((X_mini.shape[0],28,28,1))
        Y_mini = mini_batch[:, -10:]
        mini_batches.append((X_mini, Y_mini))

    # last mini batch
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-10]
        X_mini=X_mini.reshape((X_mini.shape[0],28,28,1))
        Y_mini = mini_batch[:, -10:]
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

# predict Y using X and learned parameters.
# find also accuracy
# y: (10,m) in this network
def predict(X, y, parameters):

    # take weights, biases
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]

    m = X.shape[1]
    p = np.zeros((10,m))

    # Forward pass
    forward_result = forward_prop(X, y, W1, b1,W2, b2, W3, b3, W4, b4)
    output = forward_result["A6"]

    # convert probabilities to 0/1 predictions, find accuracy
    preds= np.argmax(output,axis=0)
    y_true = np.argmax(y,axis=0)
    accuracy= np.mean(np.equal(preds, y_true))
    return preds,accuracy

"""### Model build"""

# complete forward propagation
def forward_prop(X, Y, W1, b1, W2, b2, W3, b3, W4, b4):

    # convolution
    A1, cache1 = conv_activation_forward(X, W1, b1)

    # pooling
    A2, cache2 = pool_forward(A1)

    # convolution
    A3, cache3 = conv_activation_forward(A2, W2, b2)

    # pooling
    A4, cache4 = pool_forward(A3)

    # flatten
    A4=flatten_2D_output(A4)

    # FC
    A5, cache5 = linear_activation_forward(A4, W3, b3, activation="relu")

    # FC
    A6, cache6 = linear_activation_forward(A5, W4, b4, activation="softmax")

    # Compute cost
    cost = compute_crossentropy_cost(A6,Y)

    forward_result = {
        "cost":cost,
        "A6":A6,
        "cache1":cache1,
        "cache2":cache2,
        "cache3":cache3,
        "cache4":cache4,
        "cache5":cache5,
        "cache6":cache6,
    }

    return forward_result

# complete backward propagation
def backward_prop(forward_result,Y):
    A6 = forward_result["A6"]
    cache1 = forward_result["cache1"]
    cache2 = forward_result["cache2"]
    cache3 = forward_result["cache3"]
    cache4 = forward_result["cache4"]
    cache5 = forward_result["cache5"]
    cache6 = forward_result["cache6"]

    # find dZ6
    dZ6 = softmax_crossentropy_backward(A6, Y)

    # FC
    dA5, dW4, db4 = linear_backward(dZ6, cache6[0])
    dA4, dW3, db3 = linear_activation_backward(dA5, cache5)

    # Reshape vector results to matrix results
    dA4 = deflatten_to_2D(dA4)

    # Pooling
    dA3 = pool_backward(dA4, cache4)

    # Conv
    dA2, dW2, db2 = conv_activation_backward(dA3, cache3)

    # Pooling
    dA1 = pool_backward(dA2, cache2)

    # Conv
    dA0, dW1, db1 = conv_activation_backward(dA1, cache1)

    # return  grads
    grads={}
    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2
    grads['dW3'] = dW3
    grads['db3'] = db3
    grads['dW4'] = dW4
    grads['db4'] = db4

    return grads

# train network using given hyperparams.
# X_test, y_test are used for calculating test accuracy after each epoch
def train_model(X, y, X_test, y_test, learning_rate = 0.01, num_epochs = 15, batch_size=128):

    # set seed for reproducebility
    np.random.seed(1)
    costs = []    # store minibatch costs
    m = X.shape[1]  # number of training examples

    # initialize params
    parameters = initialize_parameters()
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]

    cost=None

    # iterate over training set num_epochs times
    for i in range(num_epochs):

        # create minibatches randomly
        mini_batches = create_mini_batches(X, y, batch_size)
        counter=1 # counter for minibatch number

        for mini_batch in mini_batches:
            start_time = time.time()
            X_mini, y_mini = mini_batch

            # forward propagation
            forward_result=forward_prop(X_mini, y_mini.T, W1, b1,W2, b2, W3, b3, W4, b4)
            cost=forward_result["cost"]
            costs.append(cost)

            # backward propagation
            grads=backward_prop(forward_result,y_mini.T)

            # update params
            parameters = update_parameters(parameters, grads, learning_rate)

            # set W1, b1, W2, b2
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]
            W4 = parameters["W4"]
            b4 = parameters["b4"]

            print(counter,cost,time.time()-start_time)
            counter+=1

        print("===Epoch "+ str(i+1) +" ended=====")
        preds,accuracy=predict(X_test,y_test.T,parameters)
        print("Test accuracy: "  + str(accuracy))
        print("================")

        if accuracy>0.9:
            break

    print("== Training phase ended.==")
    return parameters

def evaluate_model(x_train, y_train, x_test, y_test, parameters):
    # train set accuracy
    preds,accuracy=predict(x_train,y_train.T,parameters)
    print("Train set accuracy: ", accuracy)

    # test set accuracy
    preds,accuracy=predict(x_test,y_test.T,parameters)
    print("Test set accuracy: ", accuracy)

def save_parameters(parameters):
    with open('learned_weights.pkl', 'wb') as f:
        pickle.dump(parameters, f)

def load_parameters():
    with open('learned_weights.pkl', 'rb') as f:
        parameters = pickle.load(f)
    return parameters

def evaluate_model_v2(parameters):
    # load dataset
    x_train,x_test,y_train,y_test=load_dataset()

    # train set accuracy
    print("Calculating train set accuracy...")
    preds,accuracy=predict(x_train,y_train.T,parameters)
    print("Train set accuracy: ", accuracy)

    # test set accuracy
    print("Calculating test set accuracy...")
    preds,accuracy=predict(x_test,y_test.T,parameters)
    print("Test set accuracy: ", accuracy)

# run model with given hyperparameters.
def run_model():

    # load dataset
    x_train,x_test,y_train,y_test=load_dataset()

    # train model with SGD (no momentum)
    parameters=train_model(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_epochs = 10, batch_size=128)

    # evaluate model
    evaluate_model(x_train, y_train, x_test, y_test, parameters)

    # save params
    save_parameters(parameters)
    return parameters

if __name__ == "__main__":
    parameters=run_model()
