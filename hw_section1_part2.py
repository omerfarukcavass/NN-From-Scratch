
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pickle

# Model parameters
num_classes = 10
input_shape = (28, 28, 1)

def load_dataset():
    # create train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range as usual
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Convert image shapes to (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # one hot encoding for class labels using keras function
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train,x_test,y_train,y_test

def build_model():
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(4, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )
    return model

# initialize weights and biases using normal dist.
# (Same weights as NN from scratch)
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

def set_initial_params(parameters, model):

    #model.layers[0].get_weights()[0].shape
    #model.layers[0].get_weights()[1].shape
    #model.layers[2].get_weights()[0].shape
    #model.layers[2].get_weights()[1].shape
    #model.layers[5].get_weights()[0].shape
    #model.layers[5].get_weights()[1].shape
    #model.layers[6].get_weights()[0].shape
    #model.layers[6].get_weights()[1].shape

    model.layers[0].set_weights([parameters["W1"],parameters["b1"].reshape((4,))])
    model.layers[2].set_weights([parameters["W2"],parameters["b2"].reshape((8,))])
    model.layers[5].set_weights([parameters["W3"].T,parameters["b3"].reshape((128,))])
    model.layers[6].set_weights([parameters["W4"].T,parameters["b4"].reshape((10,))])

def get_parameters(model):
    parameters = {"W1": model.layers[0].get_weights()[0],
                  "b1": model.layers[0].get_weights()[1],
                  "W2": model.layers[2].get_weights()[0],
                  "b2": model.layers[2].get_weights()[1],
                  "W3": model.layers[5].get_weights()[0],
                  "b3": model.layers[5].get_weights()[1],
                  "W4": model.layers[6].get_weights()[0],
                  "b4": model.layers[6].get_weights()[1],
                 }
    return parameters

def set_parameters(model,parameters):
    model.layers[0].set_weights([parameters["W1"],parameters["b1"]])
    model.layers[2].set_weights([parameters["W2"],parameters["b2"]])
    model.layers[5].set_weights([parameters["W3"],parameters["b3"]])
    model.layers[6].set_weights([parameters["W4"],parameters["b4"]])

def save_parameters(parameters):
    with open('learned_weights_library.pkl', 'wb') as f:
        pickle.dump(parameters, f)

def load_parameters():
    with open('learned_weights_library.pkl', 'rb') as f:
        parameters = pickle.load(f)
    return parameters

def evaluate_model(parameters):
    # build model
    x_train,x_test,y_train,y_test=load_dataset()
    model=build_model()
    set_parameters(model,parameters)

    opt = keras.optimizers.SGD(learning_rate=0.01) # default SGD: no momentum
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # evaluate model
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Train accuracy:", score[1])
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy:", score[1])

def train_model(x_train, y_train, model):
    batch_size = 128
    epochs = 2

    opt = keras.optimizers.SGD(learning_rate=0.01) # default SGD: no momentum
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

def run_model():
    # set seed for reproducibility
    np.random.seed(1)
    tf.random.set_seed(1)

    # load dataset
    x_train,x_test,y_train,y_test=load_dataset()

    # build model structure
    model=build_model()
    print(model.summary())

    # set initial parameters
    parameters = initialize_parameters()
    set_initial_params(parameters, model)

    # train model with SGD
    train_model(x_train,y_train,model)

    # evaluate model
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Train accuracy:", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy:", score[1])

    # save parameters
    parameters = get_parameters(model)
    save_parameters(parameters)

    return model

# run CNN model
if __name__ == "__main__":
    model = run_model()
