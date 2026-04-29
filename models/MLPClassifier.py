import numpy as np
from models.blocks import Linear, ReLU, Sigmoid, Tanh

class MLPClassifier:
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim, activation_strategy="relu"):
        self.fc1 = Linear(input_dim=input_dim, output_dim=hidden_dim_1)
        self.fc2 = Linear(input_dim=hidden_dim_1, output_dim=hidden_dim_2)
        self.fc3 = Linear(input_dim=hidden_dim_2, output_dim=hidden_dim_3)
        self.fc4 = Linear(input_dim=hidden_dim_3, output_dim=output_dim)

        self.activation_strategy = activation_strategy

        if self.activation_strategy == "relu":
            self.acti1, self.acti2, self.acti3 = ReLU(), ReLU(), ReLU()
        elif self.activation_strategy == "sigmoid":
            self.acti1, self.acti2, self.acti3 = Sigmoid(), Sigmoid(), Sigmoid()
        elif self.activation_strategy == "tanh":
            self.acti1, self.acti2, self.acti3 = Tanh(), Tanh(), Tanh()
        else:
            raise NotImplementedError(f'Activation function "{self.activation_strategy}" is not supported')
        
        self.modules = [self.fc4, self.acti3, self.fc3, self.acti2, self.fc2, self.acti1, self.fc1]

    def forward(self, x):
        z1 = self.fc1.forward(x)
        z1 = self.acti1.forward(z1)
        z2 = self.fc2.forward(z1)
        z2 = self.acti2.forward(z2)
        z3 = self.fc3.forward(z2)
        z3 = self.acti3.forward(z3)
        logits = self.fc4.forward(z3)

        return logits

    def backward(self, down_grad):
        for m in self.modules:
            down_grad = m.backward(down_grad)

        return down_grad
    
    def update(self, lr=0.01, wd=1e-4):
        for module in self.modules:
            if module.learnable:
                module.update(lr, wd)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
        
    def save_weights(self, filepath):
        weights = {
            'fc1_weight': self.fc1.weight,
            'fc1_bias': self.fc1.bias,
            'fc2_weight': self.fc2.weight,
            'fc2_bias': self.fc2.bias,
            'fc3_weight': self.fc3.weight,
            'fc3_bias': self.fc3.bias,
            'fc4_weight': self.fc4.weight,
            'fc4_bias': self.fc4.bias,
        }
        np.savez(filepath, **weights)
        print(f"Model saved to {filepath}")
    
    def load_weights(self, filepath):
        data = np.load(filepath)
        self.fc1.weight = data['fc1_weight']
        self.fc1.bias = data['fc1_bias']
        self.fc2.weight = data['fc2_weight']
        self.fc2.bias = data['fc2_bias']
        self.fc3.weight = data['fc3_weight']
        self.fc3.bias = data['fc3_bias']
        self.fc4.weight = data['fc4_weight']
        self.fc4.bias = data['fc4_bias']
        print(f"Model loaded from {filepath}")