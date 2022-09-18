# Neural Network Implementation & ReLU Decision Boundry

## Implementing a Network from Scratch

### Description

In this project, we will implement and train a convolutional neural network from scratch. We use the MNIST dataset. We implement the following network: 

![Network Architecture](https://github.com/omerfarukcavass/NN-From-Scratch/blob/main/network-arch.png)

The optimizer I used is Stochastic Gradient Descent with no momentum. 
First, I prefered to start with a simple optimizer and use more complicated ones if required. 
But I obtained quite good results even with this optimizer so I didnt switch to another one. 
Learning rate is 0.01 and batch size is 128. After 2 epoches, the network achieved test accuracy over 90%. 
So I stopped after two epoches since it takes about 80 minutes for each epoch in my laptop.

<br>

For matrix operations, I used numpy [1] library. For visualization needs, especially for the second section, I used matplotlib [2]. 
To load MNIST dataset, I used Keras library[3]. I also used time and pickle libraries from Python base libraries. 
To create dataframe in the second section , I used pandas library [4]. 
Also, to better understand how CNNs work, to design my coding flow in implementing neural network better and to understand how they approach the problem, 
I benefited from several resources [5],[6],[7],[8],[10]. I used Desmos [9] to draw lines in section 2.

### Sanity Check: 

I used Keras [3] library to implement the same network. 
I used learning rate of 0.01 and batch size of 128 as before. I also trained 2 epoches as in my implementation before.
I used same initializations in both implementations. In my own implementation,
I obtained 0.9182 test set accuracy and 0.91245 train set accuracy after two epoches. In the library implementation, 
I obtained 0.921 test accuracy and 0.913 train accuracy.
The difference is quite small and I think it results from the shuffling in minibatches 
because I used same hyperparameters and same initial weightes, biases. 
The difference in minibatches could cause differences as expected since it affects the gradients and therefore the weight updates.

## Decision Boundary of a Neural Network with ReLU







