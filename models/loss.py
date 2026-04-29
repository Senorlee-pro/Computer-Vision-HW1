import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets
        batch_size = logits.shape[0]
    
        # log_softmax = logits - log(sum(exp(logits)))
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        log_sum_exp = np.log(np.sum(exp_logits, axis=1))
        log_probs = shifted_logits - log_sum_exp.reshape(-1, 1)

        loss = -np.mean(log_probs[np.arange(batch_size), targets])
        self.log_probs = log_probs
        
        return loss
    
    def backward(self):
        # return ∂L/∂logits
        # grad = probs - one_hot(targets)
        
        batch_size = self.logits.shape[0]
        num_classes = self.logits.shape[1]
        
        # probability
        shifted_logits = self.logits - np.max(self.logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), self.targets] = 1
        
        grad = (probs - one_hot) / batch_size
        
        return grad
