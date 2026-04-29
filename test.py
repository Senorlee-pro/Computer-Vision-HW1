import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from models.MLPClassifier import MLPClassifier
from dataset.dataset import Dataloader
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return cm

def visualize_misclassified_images(model, test_loader, num_images=16):
    misclassified_images = []
    misclassified_true_labels = []
    misclassified_pred_labels = []
    for image_batch, label_batch in test_loader:
        logits = model.forward(image_batch)
        predictions = np.argmax(logits, axis=1)

        errors = predictions != label_batch
        if np.any(errors):
            error_images = image_batch[errors]
            error_true = label_batch[errors]
            error_pred = predictions[errors]
            
            for img, true_l, pred_l in zip(error_images, error_true, error_pred):
                misclassified_images.append(img)
                misclassified_true_labels.append(true_l)
                misclassified_pred_labels.append(pred_l)
                
            if len(misclassified_images) >= num_images:
                break
    
    num_to_show = min(num_images, len(misclassified_images))
    if num_to_show == 0:
        return

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    cols = min(4, int(np.ceil(np.sqrt(num_to_show))))
    rows = int(np.ceil(num_to_show / cols))
    
    plt.figure(figsize=(4*cols, 3*rows))
    
    for i in range(num_to_show):
        plt.subplot(rows, cols, i + 1)
        
        img = misclassified_images[i].reshape(28, 28)
        
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {class_names[misclassified_true_labels[i]]}\nPred: {class_names[misclassified_pred_labels[i]]}',
                 color='red', fontsize=10)
        plt.axis('off')
    
    plt.suptitle(f'Misclassified Images (Total: {len(misclassified_images)})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def test(weights_path, batch_size=128, visualize_errors=True, num_errors_to_show=16, show_confusion_matrix=True):
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True
    )
    test_loader = Dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(
        input_dim=784, 
        hidden_dim_1=512, 
        hidden_dim_2=256,
        hidden_dim_3=128, 
        output_dim=10, 
        activation_strategy="relu"
    )

    model.load_weights(weights_path)

    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    for image_batch, label_batch in test_loader:
        logits = model.forward(image_batch)
        predictions = np.argmax(logits, axis=1)
        
        total_correct += np.sum(predictions == label_batch)
        total_samples += label_batch.shape[0]
        
        all_predictions.extend(predictions)
        all_labels.extend(label_batch)

    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    if show_confusion_matrix:
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        plot_confusion_matrix(np.array(all_labels), np.array(all_predictions), class_names)
    
    if visualize_errors:
        test_loader_for_vis = Dataloader(test_dataset, batch_size=batch_size, shuffle=False)
        visualize_misclassified_images(model, test_loader_for_vis, num_images=num_errors_to_show)

if __name__ == "__main__":
    weight_file = "best_model.npz"
    test(weight_file, visualize_errors=True, num_errors_to_show=4, show_confusion_matrix=True)