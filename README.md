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

In this section, we study if it is possible to obtain a non-linear decision boundary with ReLU acti- vation function.The training data is as follows: 
![datapoints](https://github.com/omerfarukcavass/NN-From-Scratch/blob/main/data-points.png)

I trained a neural network with one hidden layer. The hidden layer has 30 neurons with ReLU activation function. 
The output layer is one neuron with sigmoid activation function for binary classification. 
Although it would be better to add one more hidden layer, I intentionally designed the network this way because my explanation in following question fit 
this architecture quite well. To plot the decision boundry of the network, I generated 10 thousand data points within the range [-3,3] , 
then predict them with this model. The decison boundry is nonlinear with its shape like a polygon as seen in figure below. 
The straight line behavior of the sides comes from the ReLU activation function properties. 
In our network, there is 30 neurons and we can make the decision boundry more smooth by increasing the number of neurons arbitrarily. 
Here, we can see that although ReLU is a piecewise linear function, nonlinear functions can also be approximated when many of them are combined.

![decboundry](https://github.com/omerfarukcavass/NN-From-Scratch/blob/main/decision-boundry.png)


We have seen it is possible to obtain a non-linear decision boundary with ReLU activation function. 
 I prepared an example in two dimension similar to the previous question, where we make a binary clasification in 2-D space,
 but the same idea could be applied in other examples, also in predicting a scalar/vectoral value in regression problems.
In the figure below, we have four lines enclosing a square region. Lets say we have a classification problem in 2-D space,
 in which all points within this square region belong to class 1, and all others are in class 2. 
A simple neural network with one hidden layer that have four neurons is sufficient to handle this problem.

![relu-decision](https://github.com/omerfarukcavass/NN-From-Scratch/blob/main/relu-decision-boundry.png)

The four lines in this graph corresponds to four neurons in hidden layer. 
The coefficients of x and y correspond to weights and the constant term corresponds to bias of a neuron. Each line splits the 2D space into two halves. 
So, each neuron determines which side a data point locates in this space. When left hand side of the all four equations are less than or equal to right hand side
 of each one, we obtain the square region above as seen in figure below. So, if all neurons get value less than or equal zero in linear output, 
that data point should belong to class 1. If any of them has positive value, then it belongs to class 0.

![relu-decision-2](https://github.com/omerfarukcavass/NN-From-Scratch/blob/main/relu-decision-2.png) 


Since ReLu function takes value 0 when the input is less than zero, data points that has zero activation in all four neurons belong to class 1 while data points 
that has at least one positve activation will be in class 0. To construct such “AND” relation, we connect four neurons to one neuron in last layer.
 
z=𝛽 +𝛽𝑥 +𝛽𝑥 +𝛽𝑥 +𝛽𝑥

<br>

In the linear activation of the last layer, z, assume all weights are positive and the bias is negative. We also know that xi are all nonnegative value due to ReLu activation. Then this equation will output negative value if all xi are zero, and positive value when ∑𝑖>0 𝛽𝑖𝑥𝑖 is greater than Bo. If the bias term is sufficiently small (or weights are sufficiently large), then it practically means that if any of the xi is positive, z takes positive value. If z takes negative value, the sigmoid function will give less than 0.5 (class 1) and if z takes positive value, it will give greater than 0.5 (class 0). Thus, we obtained a classifier with four ReLU activated neurons and one sigmoid activated neuron. Also, if we increase number of neurons in hidden layer arbitrarily, we could even approximate a circular decision boundry. Since the problem we had was a classification problem, I used this network structure. Yet, the ReLU activation function can also be used for regression problems. Finally, I didnt use any resource for this example, this example come to my mind when I see the decision boundry in previous question. So, there is no reference for this part.

### References: 

[1] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020)

[2] J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007

[3] Chollet, F., & others. (2015). Keras. GitHub. Retrieved from https://github.com/fchollet/keras 

[4] McKinney, W., & others. (2010). Data structures for statistical computing in python. In
Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51–56).

[5] https://androidkt.com/implement-softmax-and-cross-entropy-in-python-and-pytorch/

[6] https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross- entropy-loss-ffceefc081d1

[7] https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/ [8] https://pylessons.com/Deep-neural-networks-part3

[9] https://www.desmos.com/calculator?lang=tr

[10] https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks- 260c2de0a050/





