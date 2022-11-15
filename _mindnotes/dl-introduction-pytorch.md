---
title: "Introduction - PyTorch"
topic: dl-basics
collection: dl-basics
permalink: /mindnotes/dl-introduction-pytorch
---

<img src="logo_cmmf.png"
     alt="Markdown Monster icon"
     style="float: right" />
# MindNote - Deep Learning - Introduction PyTorch

**Author: Christian M.M. Frey**  
**E-Mail: <christianmaxmike@gmail.com>**

---

### Introduction PyTorch

---

## What is PyTorch?
It's a python-based scientific computing package targeted at two sets of audiences:
* a replacement for NumPy to use the power of GPUs
* a deep learning research platform that provides maximum flexibility and speed

## Pytorch introduction

Pytorch has been established to be one of the most optimized high-performance tensor library for computation of deep learning tasks on GPUs and CPUs. Pytorch is a library based on Python and the Torch tool provided by Facebook's Artificial Intelligence Research group. Pytorch provides a large collection of modules for computation with three of them being the very prominent:

* **Autograd**:  automatic differentiation of tensors. A recorder class remembers the operations and retrieves operations via backward() to compute the gradients. 
* **Optim**: optimization techniques that can be used to minimize the error function for a model. 
* **NN**: neural network library providing functions to automate the layers, activation functions, loss functions and optimization functions

In this tutorial, we make our first steps in pytorch and show how we can create tensors and also how we can execute basic operations on them. 

#### PyTorch Installation
i) Open the Anaconda navigator and go to the 'Environments' page

ii) Select your preferred environment, search for 'pytorch' and install it Alternatively: open the terminal, make sure your environment (`source activate my_env`) is selected and type the following:
> `conda install pytorch`

iii) Launch jupyter and open a notebook of your choice

iv) Type the following command to check whether the PyTorch library is installed and check the version
```python
import torch
torch.__version__
```

## Getting started

#### Load dependencies


```python
import torch
torch.__version__
```




    '1.9.0'



#### How to create tensors?

Create an empty Tensors of size (5,5), i.e., a 2-dimensional matrix with 5 rows and 5 columns.


```python
torch.empty(5,5)
```




    tensor([[0.0000e+00, 0.0000e+00, 1.4013e-45, 0.0000e+00, 1.4013e-45],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 2.8026e-45, 0.0000e+00],
            [9.4885e-39, 3.4438e-41, 1.4013e-45, 0.0000e+00, 1.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, 2.8026e-45, 0.0000e+00],
            [9.4886e-39, 3.4438e-41, 1.4013e-45, 0.0000e+00, 1.0000e+00]])



Create a tensor from a python list [[20, 30, 40], [90, 60, 70]], i.e., a 2x3 matrix wit given entries.


```python
torch.FloatTensor([[20, 30, 40], [90, 60, 70]])
```




    tensor([[20., 30., 40.],
            [90., 60., 70.]])



Create a random tensor (torch.randn()) of various sizes and print them on the console


```python
# Construct a random matrix
a = torch.Tensor(3,3)
print (a)
print (torch.randn(4,4))
```

    tensor([[2.0033e-24, 4.5873e-41, 2.0183e-24],
            [4.5873e-41, 2.0183e-24, 4.5873e-41],
            [1.6147e-24, 4.5873e-41, 2.0162e-24]])
    tensor([[-1.0667, -2.2027,  0.9573, -1.3560],
            [ 0.7074,  0.5609, -0.4915, -0.1674],
            [-0.0229, -1.0087,  0.8788,  0.1121],
            [-0.4559, -0.0781,  0.9201,  0.2624]])


Print the size of a matrix by using the function tensor.size()


```python
print(a)
print(a.size())
```

    tensor([[2.0033e-24, 4.5873e-41, 2.0183e-24],
            [4.5873e-41, 2.0183e-24, 4.5873e-41],
            [1.6147e-24, 4.5873e-41, 2.0162e-24]])
    torch.Size([3, 3])


Special matrices/tensors: Ones Tensor ; Zeros Tensor ; Identity Tensor


```python
# Create a tensor with '0' entries of size (5,3)
torch.zeros(5,3)
# Create a tensor with '1' entries of size (2,2)
torch.ones(2,2)
# Create the identity matrix (matrix having '1' on the diagonal, else '0' on the off-diagonals)
torch.eye(6)
```




    tensor([[1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]])



#### Operation on Tensors
There are multiple syntaxes for operations which we are learning in the following.
First, let's create two random tensor having the same size.


```python
x = torch.rand(3,3)
y = torch.rand(3,3)
```

#### Add tensors
There are three different ways on how to add to tensors:


```python
# Alt 1: using the '+' operator
z = x + y
print ("z:", z, "\nx: ", x, "\ny:", y)
```

    z: tensor([[1.3688, 0.7951, 1.4088],
            [0.4504, 0.5946, 1.3274],
            [1.7864, 1.1904, 1.1175]]) 
    x:  tensor([[0.4275, 0.4471, 0.9603],
            [0.0266, 0.0864, 0.9078],
            [0.8902, 0.9344, 0.8658]]) 
    y: tensor([[0.9413, 0.3480, 0.4485],
            [0.4239, 0.5082, 0.4195],
            [0.8963, 0.2560, 0.2518]])



```python
# Alt 2: using pytorch's built-in function torch.add(.,.)
torch.add(x,y)
```




    tensor([[1.3688, 0.7951, 1.4088],
            [0.4504, 0.5946, 1.3274],
            [1.7864, 1.1904, 1.1175]])



Inplace operations are post-fixed with an "_"


```python
# Alt 3: calling the tensor's add_ function.
x.add_(y) # inplace add-operation
```




    tensor([[1.3688, 0.7951, 1.4088],
            [0.4504, 0.5946, 1.3274],
            [1.7864, 1.1904, 1.1175]])




```python
print(x)
```

    tensor([[1.3688, 0.7951, 1.4088],
            [0.4504, 0.5946, 1.3274],
            [1.7864, 1.1904, 1.1175]])


#### Multiply two tensors
For the multiplicaiton of a tensor with a scalar, there are again the three different variants from above.


```python
# scalar multiplication
# Alternative 1  
print (x * 5)           
# Alternative 2
print (torch.mul(x,5))  
# Alternative 3
print (x.mul_(5))       
```

    tensor([[ 855.4833,  496.9533,  880.5090],
            [ 281.5063,  371.6346,  829.6192],
            [1116.5049,  744.0276,  698.4658]])
    tensor([[ 855.4833,  496.9533,  880.5090],
            [ 281.5063,  371.6346,  829.6192],
            [1116.5049,  744.0276,  698.4658]])
    tensor([[ 855.4833,  496.9533,  880.5090],
            [ 281.5063,  371.6346,  829.6192],
            [1116.5049,  744.0276,  698.4658]])


#### Transpose
Transposing a (n,m)-matrix, result in its (m,n) tranposed version (=switching rows and cols)


```python
# Alternative 1
x.t()   
# Alternative 2
x.transpose(1,0)
```




    tensor([[ 855.4833,  281.5063, 1116.5049],
            [ 496.9533,  371.6346,  744.0276],
            [ 880.5090,  829.6192,  698.4658]])



#### Concatenation of Tensors
PyTorch provides already a function to concatenate two tensors along a specific dimension.
Let's create two random tensors:


```python
a = torch.rand(4,3)
b = torch.rand(4,3)
```


```python
# Concatenating along the '0'-th dimension (appending as rows)
cat1 = torch.cat((a,b), 0)
print (cat1)

# Concatenating along the '1'-st dimension (appending as cols)
cat2 = torch.cat((a,b), 1)
print (cat2)
```

    tensor([[0.7584, 0.3989, 0.1433],
            [0.8016, 0.7306, 0.8424],
            [0.1678, 0.7775, 0.0676],
            [0.7486, 0.8901, 0.6153],
            [0.7705, 0.9654, 0.1987],
            [0.6932, 0.8022, 0.1134],
            [0.1097, 0.6701, 0.1871],
            [0.3566, 0.5516, 0.6091]])
    tensor([[0.7584, 0.3989, 0.1433, 0.7705, 0.9654, 0.1987],
            [0.8016, 0.7306, 0.8424, 0.6932, 0.8022, 0.1134],
            [0.1678, 0.7775, 0.0676, 0.1097, 0.6701, 0.1871],
            [0.7486, 0.8901, 0.6153, 0.3566, 0.5516, 0.6091]])


#### Slicing of Tensors
Slicing is similar to the slicing operations which we know form numpy:


```python
# get the 3rd column
a[:, 2]
```




    tensor([0.1433, 0.8424, 0.0676, 0.6153])




```python
# get the 3rd row
a[2, :]
```




    tensor([0.1678, 0.7775, 0.0676])



#### Resize a Tensor via *view()* command
Often it might come in handy two reorder the entries within a tensor. 
For that pyTorch provides the .view() operation:


```python
# Create a random tensor of size (4,4)
x = torch.rand(4,4)

# Resize the entries such that there is only one row with all the entries.
y = x.view(-1,16)
print (y)
print (x.flatten())  #  <<-- Another way to flatten an array

# Resize the entries such that in each row there are 8 entries resulting in a (2,8) matrix
z1 = x.view(2, 8)
z2 = x.view(-1, 8) #  <-- -1 can be used whenever the remaining entries are meant

print(x.size(), y.size(), z1.size())
```

    tensor([[0.2188, 0.3941, 0.7353, 0.5304, 0.1330, 0.0808, 0.2885, 0.9352, 0.2935,
             0.0148, 0.5019, 0.8725, 0.9420, 0.8920, 0.3253, 0.5555]])
    tensor([0.2188, 0.3941, 0.7353, 0.5304, 0.1330, 0.0808, 0.2885, 0.9352, 0.2935,
            0.0148, 0.5019, 0.8725, 0.9420, 0.8920, 0.3253, 0.5555])
    torch.Size([4, 4]) torch.Size([1, 16]) torch.Size([2, 8])


#### NumPy Bridge
The torch tensor and numpy array will share their underlying memory locations. Therefore, changing one will change the other.


```python
# Create a tensor with '1' entries of size (5,4)
a = torch.ones(5,4)
```


```python
print(a)
print (type(a)) # <<-- let's check a's type
```

    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    <class 'torch.Tensor'>



```python
a_np = a.numpy()
print (type(a_np))  # <<-- checking the type again
```

    <class 'numpy.ndarray'>


Switching a tensor to its numpy representation, we can of course then apply the methods we know from numpy on the data.


```python
# Torch Tensor -> NumPy array
b = a.numpy()
b = b.reshape(4,-1)
```


```python
print(b)
```

    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]


Switching back from numpy to pyTorch, we can use the function torch.from_numpy().


```python
# NumPy Array -> Torch Tensor
c = torch.from_numpy(b)
print(c)
```

    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])



```python
import numpy as np
a = np.random.rand(3,4)
b = torch.from_numpy(a)
print(b)
```

    tensor([[0.6912, 0.7155, 0.3031, 0.4622],
            [0.6994, 0.5682, 0.2444, 0.5003],
            [0.2115, 0.5563, 0.7044, 0.1517]], dtype=torch.float64)


### Dynamic Computational Graph
The dynamic computational graph is combined of gradient enabled tensors (variables) along with functions (operations). The flow of data and the operations applied to the data are defined at runtime.

Every variable object has several attributes:
* **Data**: data of the variable
* **requires_grad**: if true, a tracking of the operation history and is started and a backward graph for gradient calculation is formed
* **grad**: holds the value of the gradient, e.g., if you call out.backward() for some variable out that involved a variable x in its calculation then x.grad will hold $\partial out/ \partial x$.
* **grad_fn**: backward function being used to calculate the gradient
* **is_leaf**: indicating whether the node is a leaf node. A node is a leaf if i) it was initialized explicitly by some function; ii) is is created after operations on tensors which all have *required_grad=False*; iii) it is created by calling *.detach()* method on some tensor


```python
# Creating the graph
x = torch.tensor(1.0, requires_grad = True)
y = torch.tensor(2.0)
z = x * y

# Displaying
for i, name in zip([x, y, z], "xyz"):
    print(f"{name}\n\
        data: {i.data}\n\
        requires_grad: {i.requires_grad}\n\
        grad: {i.grad}\n\
        grad_fn: {i.grad_fn}\n\
        is_leaf: {i.is_leaf}\n")
```

    x
            data: 1.0
            requires_grad: True
            grad: None
            grad_fn: None
            is_leaf: True
    
    y
            data: 2.0
            requires_grad: False
            grad: None
            grad_fn: None
            is_leaf: True
    
    z
            data: 2.0
            requires_grad: True
            grad: None
            grad_fn: <MulBackward0 object at 0x7fe01b8f1630>
            is_leaf: False
    


    /Users/ChrisMaxMike/anaconda3/envs/py36_torch/lib/python3.6/site-packages/ipykernel_launcher.py:13: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information.
      del sys.path[0]


### Autograd mechanism : Automatic differentitation

*torch.Tensor* is the central class of the package. If you set its attribute *.requires_grad* as *True*, it starts to track all operations on it. When you finish your computation you can call *.backward()* and have all the gradients computed automatically. The gradient for this tensor will be accumulated into *.grad* attribute


```python
from torch.autograd import Variable
```


```python
x = Variable(torch.ones(4,3), requires_grad=True)
print (x)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], requires_grad=True)



```python
y = x + 5
print(y)
```

    tensor([[6., 6., 6.],
            [6., 6., 6.],
            [6., 6., 6.],
            [6., 6., 6.]], grad_fn=<AddBackward0>)



```python
print(y.grad_fn)
```

    <AddBackward0 object at 0x7fe01b8ff898>



```python
z = y * y * 5
out = z.mean()
print(z, out)
```

    tensor([[180., 180., 180.],
            [180., 180., 180.],
            [180., 180., 180.],
            [180., 180., 180.]], grad_fn=<MulBackward0>) tensor(180., grad_fn=<MeanBackward0>)



```python
out.backward()
```


```python
print(x.grad)
```

    tensor([[5., 5., 5.],
            [5., 5., 5.],
            [5., 5., 5.],
            [5., 5., 5.]])


Why 5? With 'out' we have that $out = \frac{1}{4\cdot3} \sum_{i} z_i$, where $z_i = 5 \cdot (x+5)^2$. Therefore, $\frac{\partial out}{\partial x_i} = \frac{10}{12} (x_i + 5)$. Hence, $\frac{\partial out}{\partial x_i} \Bigr\rvert_{x_i = 1} = \frac{60}{12} = 5$

#### Notes about the Backward function
*Backward()* is used for calculating the gradient by passing it's argument (default: 1x1 unit tensor) through the backward graph all the way up to every leaef node traceable from the calling root tensor. The calculated gradients are then stored in *.grad* of every leaf node.

Hence, a tensor is automatically passed as .backward(torch.tensor(1.0)). The dimension of tensor passed into *.backward()* must be the same as the dimension of the tenso whose gradient is being calculated. For example:


```python
%%time
x = torch.randn(2,2)
x = Variable(x, requires_grad = True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
    
print(y.data.norm())
print(y)

gradients = torch.FloatTensor(torch.rand(2, 2))
y.backward(gradients)
print(x.grad)
```

    tensor(1875.4691)
    tensor([[1350.1831,  986.6542],
            [  46.8405, -847.7674]], grad_fn=<MulBackward0>)
    tensor([[ 831.6336,  324.4223],
            [ 780.7774, 1003.5716]])
    CPU times: user 5.8 ms, sys: 2.8 ms, total: 8.6 ms
    Wall time: 12.9 ms


The tensor passed into the backward function acts like weights for a weighted output of gradient. 

In the example *y* is no longer a scalar. torch.autograd count not compute the full Jacobian directly, but if we just want the vector-Jacobian product, we can simply pass a vector to the backward() function as argument.

#### Further references:
* [Pytorch - What is pytorch](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
* [Pytorch - Autograd: Automatic Differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

# End of this MindNote
