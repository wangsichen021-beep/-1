import matplotlib.pyplot as plt
from dataset import load_fashion_mnist
from layers import MLP3Hidden
from train import train
from train import evaluate_and_plot
import numpy as np
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()

    param_grid = [
        {'h1': 256, 'h2': 128, 'h3': 64, 'lr': 0.1},
        {'h1': 512, 'h2': 256, 'h3': 128, 'lr': 0.05}
    ]

    best_val_acc = 0
    final_model = None
    final_history = None

    for params in param_grid:
        print(f"\nTesting Config: {params}")
        model = MLP3Hidden(784, params['h1'], params['h2'], params['h3'], 10)
        history = train(model, (X_train, y_train, X_val, y_val), lr=params['lr'])
        
        if max(history['val_acc']) > best_val_acc:
            best_val_acc = max(history['val_acc'])
            final_model = model
            final_history = history

    evaluate_and_plot(final_model, X_test, y_test, final_history)

    W1 = final_model.fc1.W
    plt.figure(figsize=(6, 6))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(W1[:, i].reshape(28, 28), cmap='seismic')
        plt.axis('off')
    plt.suptitle("First Layer Weights Visualization")
    plt.show()

    class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

test_logits = final_model.forward(X_test)
test_preds = np.argmax(test_logits, axis=1)

incorrect_indices = np.where(test_preds != y_test)[0]
print(f"\n[错例分析] 测试集总数: {len(y_test)} | 错误数量: {len(incorrect_indices)} | 准确率: {(1 - len(incorrect_indices)/len(y_test)):.4f}")

plt.figure(figsize=(15, 6))
num_to_show = min(5, len(incorrect_indices))
selected_idx = np.random.choice(incorrect_indices, num_to_show, replace=False)

for i, idx in enumerate(selected_idx):
    plt.subplot(1, num_to_show, i + 1)
    
    img = X_test[idx].reshape(28, 28)
    
    plt.imshow(img, cmap='magma')
    
    true_label = class_names[y_test[idx]]
    pred_label = class_names[test_preds[idx]]
    
    plt.title(f"Index: {idx}\nTrue: {true_label}\nPred: {pred_label}", color='red', fontsize=10)
    plt.axis('off')

plt.suptitle("Error Analysis: Random Misclassified Samples", fontsize=14)
plt.tight_layout()
plt.show()

from collections import Counter
error_pairs = [(class_names[y_test[i]], class_names[test_preds[i]]) for i in incorrect_indices]
most_common_errors = Counter(error_pairs).most_common(5)

print("\n--- 最常混淆的类别对 (前5名) ---")
for pair, count in most_common_errors:
    print(f"真标签 [{pair[0]}] 被误判为 [{pair[1]}]: {count} 次")