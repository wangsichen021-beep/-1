from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset import load_fashion_mnist
from model import MLP 
from train import train
from train import compute_accuracy
import matplotlib.pyplot as plt
import numpy as np

X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()

hidden_dims = [64, 128]
learning_rates = [0.1, 0.05]
activations = ['relu', 'tanh']

best_global_acc = 0
best_params = {}

for hd in hidden_dims:
    for lr in learning_rates:
        for act in activations:
            print(f"\n--- Training with Hidden: {hd}, LR: {lr}, Act: {act} ---")
            model = MLP(hidden_dim=hd, activation=act)
            history = train(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=128, lr=lr)
            
            val_acc = max(history['val_acc'])
            if val_acc > best_global_acc:
                best_global_acc = val_acc
                best_params = {'hd': hd, 'lr': lr, 'act': act}

print("\n=== Best Hyperparameters ===")
print(best_params)

best_model = MLP(hidden_dim=best_params['hd'], activation=best_params['act'])
history = train(best_model, X_train, y_train, X_val, y_val, epochs=30, batch_size=128, lr=best_params['lr'])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy Curve')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('learning_curves.png')
plt.show()

test_acc = compute_accuracy(best_model, X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

logits = best_model.forward(X_test)
y_pred = np.argmax(logits, axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')
plt.savefig('confusion_matrix.png')
plt.show()


W1 = best_model.fc1.W 
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    weight_img = W1[:, i].reshape(28, 28)
    ax.imshow(weight_img, cmap='seismic') 
    ax.axis('off')
plt.suptitle('Visualization of 1st Hidden Layer Weights')
plt.show()

incorrect_indices = np.where(y_pred != y_test)[0]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    idx = incorrect_indices[i]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
    ax.axis('off')
plt.show()