
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import time

def load_dataset():
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    y_train=y_train.reshape((60,1))
    y_test=y_test.reshape((40,1))

    print("X_train:",X_train.shape)
    print("y_train:",y_train.shape)
    print("X_test:",X_test.shape)
    print("y_test:",y_test.shape)
    return X_train,y_train,X_test,y_test

def plot_datapoints(X_train, y_train):
    df=pd.DataFrame(np.hstack((X_train, y_train)))
    fig, ax = plt.subplots(figsize=(8,8))
    colors = {0:'red', 1:'green'}
    ax.scatter(df[0], df[1], c=df[2].map(colors))
    plt.show()

def build_model():
    input_shape= (2)
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(30, activation="relu"),
        #layers.Dense(15, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
    )
    print(model.summary())
    return model

def run_model(X_train, y_train, batch_size = 60, epochs = 200 ):
    np.random.seed(1)
    tf.random.set_seed(1)
     # full batch gradient descent
    model = build_model()
    opt = keras.optimizers.SGD(learning_rate=0.1) # default SGD: no momentum
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # evaluate
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return model

def generate_decision_boundry(model):
    # generate random samples
    random_points=np.random.uniform(low=-3,high=3,size=(10000,2))

    # predict samples
    df_random=pd.DataFrame(random_points)
    preds=model.predict(random_points)
    preds=(preds>0.5)*1
    df_random["preds"]=preds

    # plot predictions
    fig, ax = plt.subplots(figsize=(8,8))
    colors = {0:'red', 1:'green'}
    ax.scatter(df_random[0], df_random[1], c=df_random["preds"].map(colors))
    plt.show()

X_train,y_train,X_test,y_test=load_dataset()

# plot_datapoints(X_train,y_train)

model = run_model(X_train, y_train, batch_size = 60, epochs = 200)

# generate_decision_boundry(model)
