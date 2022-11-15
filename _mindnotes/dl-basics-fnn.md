---
title: "Perceptron"
topic: dl-basics
collection: dl-basics
permalink: /mindnotes/dl-basics-ffnn
---


<img src="logo_cmmf.png"
     alt="Markdown Monster icon"
     style="float: right" />
# MindNotes - Deep Learning - Basics

**Author: Christian M.M. Frey**  
**E-Mail: <christianmaxmike@gmail.com>**

---

## PyTorch - Feedforward Neural Network
---

---

In this tutorial we will learn how to set up a simple feed forward neural network for predicting classes of handwritten digits. Therefore, we will use the MNIST dataset containing 60.000 samples in the training set and 10.000 examples in the test data. One image of MNIST has a size of 28x28 pixel.
To get more information about the MNISTS Database, please refer to : http://yann.lecun.com/exdb/mnist/

#### Load dependencies


```python
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

#### Loading MNIST Train Dataset


```python
import torchvision.transforms as transforms
import torchvision.datasets as dsets
train_data = dsets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_data = dsets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
```

#### Define batch size and number of epochs

* **Minibatches.** The MNIST dataset contains 60.000 training samples. Therefore we want to split them up to smaller groups, called mini-batches. Later on, we will pass one at a time to the neural network on which it can learn on. 
* **Iterations.** We call the process of training on one minibatch an *iteration* (i.e. one weight update). Therefore, having 60.000 images and a mini-batch size of 100, we would have 600 iterations.
* **Epoch.** An epoch means that the whole training set has been used for training the neural network. That means, if we want to learn on the whole dataset for 10 epochs, then we have in total 10*600 = 6000 iterations.


For that purpose will first define the variables *batch_size*, *num_epochs* and *n_iters* indicating the size of the minibatches, the number of epochs and the number of iterations where the latter is dependent on the first two variables.


```python
batch_size = 100
num_epochs = 10
n_iters = int(len(train_data)*num_epochs/batch_size)
print(n_iters)
```

    6000


#### DataLoader in PyTorch
PyTorch provides some very strong tools on handling loading data and preprocessing data. The MNIST dataset set having been loaded above is of type *torch.utils.data.Dataset*. *Dataset* in PyTorch is an abstract class representing a dataset. Therefore, whenever you create a custom dataset it should inherit from *Dataset* and provide the 2 methods:
* \_\_len\_\_: returns the size of the dataset
* \_\_getitem\_\_: supporting indexings such that data[i] can be used to receive the i-th sample from the dataset

A *DataLoader* can then be used as an iterator for the dataset. For this introductory example it is sufficient to know that we can tell the DataLoader the minibatch size we would like to have and that we want the samples to be reshuffled at every epoch (For further details on the DataLoader parameters, please have a look at the API).

Therefore, we will no create 2 DataLoaders, namely *train\_loader*, *test\_loader*, where the former one is a dataloader for the training data and the latter one for the test data. We provide for each of them the size of the minibatches having been calculated above and that we also want to reshuffle the training data for each epoch. 


```python
# dataloader for training set
train_load = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
# dataloader for test set
test_load = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
```

#### Create FeedForward Neural Network
Next, we will create a class for our first neural network model. As already seen in the last tutorial, we simply have to define the modules we would like to have in our model and we have to provide a *forward* function. 

We will start with a very simple model consisting of three layers, an input layer, an hidden layer and an output layer. Hence, we use two linear modules to define the linear combination from the input layer to the hidden layer, and from the hidden layer to the output layer. For the activation function of the hidden layer we use the sigmoid function. 


```python
class FeedforwardNN (nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # first module
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # activation function
        self.sigmoid = nn.Sigmoid()
        # second module
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        return self.linear2(x)
```

#### Instantiate model
The input dimension is clearly defined by the size of the images of the MNIST datset. One image is of size 28x28 pixel making the input dimension 784. The dimension of the ouput layer is defined by the classes of the MNIST dataset. Test and play around with the dimension of the hidden layer to see how it improves or downgrades your model. 

Instantiate the model and attach the dimensions for each layer as parameters.


```python
input_dim = 28*28
hidden_dim = 100
output_dim = 10
model = FeedforwardNN(input_dim, hidden_dim, output_dim)
```

As loss function we will use the cross entropy cost function having been introduces in the lecture


```python
# Define Loss Function
criterion = nn.CrossEntropyLoss()
```

As *optimizer* we will use again the stochastic gradient descent optimizer from the last notebook


```python
# Define Optimizer
learning_rate = 1e-1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

#### Train it
Next, we will train our model in the same manner as we have learned it in the previous notebook. The logic for the training procedure is as follows: 
* We learn the model for $n$ epochs
* In each epoch, we use the training dataloader to iterate the training data to get its minibatches
* Next, we execute the learning procedure (same as has already been shown in the previous notebook
* In order to get a feedback from the sytem on how good our model is learning we will also calculate the accuracy of our model on the test data (note that the test data is not used for training). Therefore, after a certain number of *iterations*/*epochs* we would like to have a response about the accuracy. For calculating the accuracy, we use the dataloader on the test data, predict the classes with our model and calculate the number of right predictions with respect to the total size of the test set. 


```python
# Training

for epoch in range(1,num_epochs+1):
    model.train()
    for i , (x_mb, y_mb) in enumerate (train_load):
        x_mb = x_mb.view(-1, 28*28)
        optimizer.zero_grad()
        y_pred = model(x_mb)
        loss = criterion(y_pred, y_mb)
        loss.backward()
        optimizer.step()

    # Validation after n epochs (here: after each epoch)
    if epoch % 1 == 0:
        model.eval()
        correct = 0
        total = 0 
        test_mb_loss = []
        for x_test, y_test in test_load:
            x_test = x_test.view(-1,28*28)
            y_pred=model(x_test)
            loss = criterion(y_pred, y_test)
            _, max_indices = torch.max(y_pred, 1)
            total += y_test.size(0)
            correct += (max_indices == y_test).sum()
        acc = 100. * correct/total

    print ("Epoch {}:\n\tTraining loss: {}\n\tAccuracy on test data: {:.2f}".format(epoch,  loss.item(), acc))
```

    Epoch 1:
    	Training loss: 0.690066397190094
    	Accuracy on test data: 87.00
    Epoch 2:
    	Training loss: 0.5256693363189697
    	Accuracy on test data: 90.00
    Epoch 3:
    	Training loss: 0.4824374318122864
    	Accuracy on test data: 90.00
    Epoch 4:
    	Training loss: 0.4349577724933624
    	Accuracy on test data: 91.00
    Epoch 5:
    	Training loss: 0.40147295594215393
    	Accuracy on test data: 92.00
    Epoch 6:
    	Training loss: 0.39231643080711365
    	Accuracy on test data: 92.00
    Epoch 7:
    	Training loss: 0.3937968313694
    	Accuracy on test data: 92.00
    Epoch 8:
    	Training loss: 0.3628355860710144
    	Accuracy on test data: 92.00
    Epoch 9:
    	Training loss: 0.3557168245315552
    	Accuracy on test data: 93.00
    Epoch 10:
    	Training loss: 0.33277031779289246
    	Accuracy on test data: 93.00


# End of this MindNote
