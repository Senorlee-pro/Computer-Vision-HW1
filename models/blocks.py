import numpy as np

class ModuleBase:
    def __init__(self):
        self.learnable = False

    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError
    
    def zero_grad(self):
        raise NotImplementedError
    


class Linear(ModuleBase):
    def __init__(self, input_dim, output_dim):
        self.learnable = True
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.weight = np.random.randn(input_dim, output_dim) * 0.01
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weight = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.zeros((1, output_dim))

    def forward(self, x):
        if np.isnan(self.weight).any() or np.isinf(self.weight).any():
            print("weight already broken before forward!")
            exit()
        self.input = x # input dimension should be (n, input_dim)
        z = x @ self.weight + self.bias
        
        return z
    
    def backward(self, down_grad): # down_grad dimension (n, output_dim)

        self.weight_grad = (self.input.T @ down_grad)
        self.bias_grad = down_grad.sum(axis=0, keepdims=True)

        self.x_grad = down_grad @ self.weight.T
        return self.x_grad
    
    def update(self, lr, wd):
        self.weight_grad += wd * self.weight

        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad

    def zero_grad(self):
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

class ReLU(ModuleBase):
    def __init__(self):
        self.learnable = False
    
    def forward(self, x):
        self.input = x
        return np.maximum(x, 0)

    def backward(self, down_grad):
        back_grad = down_grad.copy()
        back_grad[self.input <= 0] = 0
        self.back_grad = back_grad
        return back_grad
    
    def zero_grad(self):
        self.back_grad = 0

class Sigmoid(ModuleBase):
    def __init__(self):
        self.learnable = False
    
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, down_grad):
        self.back_grad = down_grad * self.output * (1 - self.output)
        return self.back_grad
    
    def zero_grad(self):
        self.back_grad = 0


class Tanh(ModuleBase):
    def __init__(self):
        self.learnable = False
    
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, down_grad):
        self.back_grad = down_grad * (1 - self.output**2)
        return self.back_grad
    
    def zero_grad(self):
        self.back_grad = 0
