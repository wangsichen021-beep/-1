import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout, l2_reg=0.0):
        self.dW = np.dot(self.x.T, dout) + l2_reg * self.W
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self.x > 0)

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out**2)

class CrossEntropyLoss:
    def __init__(self):
        self.p = None
        self.y = None

    def forward(self, logits, y):
        m = y.shape[0]
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        self.p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.y = y
        
        corect_logprobs = -np.log(self.p[range(m), y] + 1e-9)
        loss = np.sum(corect_logprobs) / m
        return loss

    def backward(self):
        m = self.y.shape[0]
        dout = self.p.copy()
        dout[range(m), self.y] -= 1
        dout /= m
        return dout

class MLP:
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, activation='relu'):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.act1 = ReLU() if activation == 'relu' else Tanh()
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1.forward(x)
        out = self.act1.forward(out)
        out = self.fc2.forward(out)
        return out

    def backward(self, dout, l2_reg=0.0):
        dout = self.fc2.backward(dout, l2_reg)
        dout = self.act1.backward(dout)
        dout = self.fc1.backward(dout, l2_reg)

    def step(self, lr):
        self.fc1.W -= lr * self.fc1.dW
        self.fc1.b -= lr * self.fc1.db
        self.fc2.W -= lr * self.fc2.dW
        self.fc2.b -= lr * self.fc2.db

    def save_weights(self):
        return {'fc1_W': self.fc1.W.copy(), 'fc1_b': self.fc1.b.copy(),
                'fc2_W': self.fc2.W.copy(), 'fc2_b': self.fc2.b.copy()}

    def load_weights(self, weights):
        self.fc1.W = weights['fc1_W']
        self.fc1.b = weights['fc1_b']
        self.fc2.W = weights['fc2_W']
        self.fc2.b = weights['fc2_b']