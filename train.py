import numpy as np
import matplotlib.pyplot as plt
import pickle
from model import CrossEntropyLoss
def compute_accuracy(model, X, y):
    logits = model.forward(X)
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == y)

def train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=128, 
          lr=0.1, lr_decay=0.95, l2_reg=0.001):
    
    criterion = CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_weights = None
    
    num_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        for i in range(0, num_samples, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            
            l2_loss = 0.5 * l2_reg * (np.sum(model.fc1.W**2) + np.sum(model.fc2.W**2))
            loss += l2_loss
            epoch_loss += loss * X_batch.shape[0]
            
            dout = criterion.backward()
            model.backward(dout, l2_reg)
            
            model.step(lr)
            
        lr *= lr_decay
        
        train_loss = epoch_loss / num_samples
        val_acc = compute_accuracy(model, X_val, y_val)
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {lr:.5f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.save_weights()
            
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    model.load_weights(best_weights) 
    
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_weights, f)
        
    return history