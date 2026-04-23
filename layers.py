import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout, l2_reg):
        self.dW = np.dot(self.x.T, dout) + l2_reg * self.W
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)

class ReLU:
    def __init__(self): self.x = None
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    def backward(self, dout):
        return dout * (self.x > 0)

class Tanh:
    def __init__(self): self.out = None
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self, dout):
        return dout * (1 - self.out**2)

class CrossEntropyLoss:
    def __init__(self): self.p = None; self.y = None
    def forward(self, logits, y):
        m = y.shape[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.y = y
        return -np.sum(np.log(self.p[range(m), y] + 1e-9)) / m

    def backward(self):
        m = self.y.shape[0]
        dout = self.p.copy()
        dout[range(m), self.y] -= 1
        return dout / m

class MLP3Hidden:
    """具备3层隐藏层的神经网络"""
    def __init__(self, input_dim, h1, h2, h3, output_dim, activation='relu'):
        Act = ReLU if activation == 'relu' else Tanh
        self.fc1 = Linear(input_dim, h1)
        self.act1 = Act()
        self.fc2 = Linear(h1, h2)
        self.act2 = Act()
        self.fc3 = Linear(h2, h3)
        self.act3 = Act()
        self.fc4 = Linear(h3, output_dim)
        
        self.layers = [self.fc1, self.act1, self.fc2, self.act2, self.fc3, self.act3, self.fc4]

    def forward(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def backward(self, dout, l2_reg):
        dout = self.fc4.backward(dout, l2_reg)
        dout = self.act3.backward(dout)
        dout = self.fc3.backward(dout, l2_reg)
        dout = self.act2.backward(dout)
        dout = self.fc2.backward(dout, l2_reg)
        dout = self.act1.backward(dout)
        dout = self.fc1.backward(dout, l2_reg)

    def step(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

    def save_weights(self):
        return [ (l.W.copy(), l.b.copy()) for l in self.layers if isinstance(l, Linear) ]

    def load_weights(self, weights):
        linears = [l for l in self.layers if isinstance(l, Linear)]
        for i, (W, b) in enumerate(weights):
            linears[i].W, linears[i].b = W, b