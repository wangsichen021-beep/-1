import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import numpy as np
from layers import CrossEntropyLoss

def train(model, data, epochs=20, batch_size=128, lr=0.1, lr_decay=0.95, l2_reg=0.001):
    X_train, y_train, X_val, y_val = data
    criterion = CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': []}
    best_acc, best_w = 0, None

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_idx = idx[i : i + batch_size]
            bx, by = X_train[batch_idx], y_train[batch_idx]
            
            logits = model.forward(bx)
            loss = criterion.forward(logits, by)
            epoch_loss += loss
            
            model.backward(criterion.backward(), l2_reg)
            model.step(lr)
            
        lr *= lr_decay
        val_acc = np.mean(np.argmax(model.forward(X_val), axis=1) == y_val)
        history['train_loss'].append(epoch_loss / (len(X_train)//batch_size))
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_w = model.save_weights()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {history['train_loss'][-1]:.4f} - Val Acc: {val_acc:.4f}")

    model.load_weights(best_w)
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_w, f)
    return history

def evaluate_and_plot(model, X_test, y_test, history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history['train_loss']); plt.title('Loss')
    plt.subplot(1, 2, 2); plt.plot(history['val_acc']); plt.title('Accuracy')
    plt.show()

    y_pred = np.argmax(model.forward(X_test), axis=1)
    print(f"Final Test Accuracy: {np.mean(y_pred == y_test):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()