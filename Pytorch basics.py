import torch 
import numpy as np

## Tensors
## A matrix is a specific type of two-dimensional tensor, while a tensor is a broader concept that can represent any number of dimensions (including scalars, vectors, and matrices), and is often used in machine learning and physics to represent multi-dimensional data

#initializing a tensor
x = torch.empty(1)
y = torch.rand(2,2)
z = torch.zeros(2,2)
w = torch.ones(2,2)

x_np = np.array([2,3])
x_tensor = torch.tensor(x_np)
print(x_tensor)

x = torch.rand(2,2)
y = torch.rand(2,2)
z = x*y #elementwise multiplication i.e. not matrix multiplication
z2 = torch.mul(x,y) #same as above #elementwise multiplication i.e. not matrix multiplication
print(x)
print(y)
print(z)
print(z2)

a1 = x+y
a2 = torch.add(x,y)
print(a1)
print(a2)

## reshaping, flattening, view etc.
x = torch.rand(4,4)
print(x)

y = x.view(2,8)
y = x.view(-1,8) #automatically identifies the correct number of rows
print(y)

