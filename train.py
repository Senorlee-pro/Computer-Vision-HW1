from models.MLPClassifier import MLPClassifier
from dataset.dataset import Dataloader, DatasetSubset
from models.loss import CrossEntropyLoss
import numpy as np
from torchvision import datasets
import itertools
import matplotlib.pyplot as plt

def evaluate(model, loader):
    total_correct = 0
    total_samples = 0
    
    for image_batch, label_batch in loader:
        logits = model.forward(image_batch)
        preds = np.argmax(logits, axis=1)
        total_correct += np.sum(preds == label_batch)
        total_samples += label_batch.shape[0]
    
    return total_correct / total_samples

def evaluate_loss(model, loader, loss_fn):
    total_loss = 0.0
    total_samples = 0
    
    for image_batch, label_batch in loader:
        logits = model.forward(image_batch)
        l = loss_fn.forward(logits=logits, targets=label_batch)
        
        total_loss += l * label_batch.shape[0]
        total_samples += label_batch.shape[0]
    
    return total_loss / total_samples

def run(hidden_dims, epochs, lr, weight_decay, batch_size=128 ,lr_decay=0.9, activation_stategy="relu", save_weights=False):
    np.random.seed(42)

    model = MLPClassifier(input_dim=784, hidden_dim_1=hidden_dims[0], hidden_dim_2=hidden_dims[1], hidden_dim_3=hidden_dims[2], output_dim=10, activation_strategy=activation_stategy)

    full_train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True
    )

    num_total = len(full_train_dataset)
    indices = np.arange(num_total)
    np.random.shuffle(indices)

    train_indices = indices[10000:]
    val_indices = indices[:10000]

    train_subset = DatasetSubset(full_train_dataset, train_indices)
    val_subset = DatasetSubset(full_train_dataset, val_indices)

    train_dataloader = Dataloader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloader(val_subset, batch_size=batch_size, shuffle=False)

    loss = CrossEntropyLoss()

    best_val_acc = 0.0

    loss_record = []
    val_loss_record = []
    acc_record = []

    for e in range(epochs):
        current_lr = lr * (lr_decay ** e)
        num_batch = 0

        for image_batch, label_batch in train_dataloader:
            logits = model.forward(image_batch)
            
            l = loss.forward(logits=logits, targets=label_batch)
            loss_record.append(l)

            if num_batch % 100 == 0:
                print(f"Epoch:{e} Batch:{num_batch} Loss:{l}")

            grad = loss.backward()

            model.backward(down_grad=grad)

            model.update(lr=current_lr, wd=weight_decay)
            model.zero_grad()
            num_batch += 1
        val_acc = evaluate(model, val_dataloader)
        acc_record.append(val_acc)
        print(f"Epoch {e} ends, accuracy on validation set: {val_acc:.4f}")
        val_loss = evaluate_loss(model, val_dataloader, loss)
        val_loss_record.append(val_loss)

        if save_weights:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_weights("best_model.npz")
                print(f"Found better model, current best accuracy: {best_val_acc:.4f} ---")
    
    return acc_record, loss_record, val_loss_record


def grid_search():
    lrs = [0.1, 0.01]
    wds = [1e-3, 1e-4]
    h_configs = [                   
        (512, 256, 128), 
        (256, 128, 64)
    ]

    results = []
    n_runs = 5

    search_space = list(itertools.product(lrs, wds, h_configs))

    for lr, wd, h_cfg in search_space:
        print(f"Testing: LR={lr}, WD={wd}, Dims={h_cfg}...")
        
        val_accs = []
        
        for run_id in range(n_runs):
            print(f"Running for {run_id+1}/{n_runs} times: ", end=" ")
            acc = run(lr=lr, hidden_dims=h_cfg, weight_decay=wd, epochs=1)
            val_accs.append(acc)
            print(f"Acc: {acc:.4f}")
        
        mean_acc = np.mean(val_accs)
        std_acc = np.std(val_accs)
        
        results.append({
            'lr': lr,
            'wd': wd,
            'hidden_dims': h_cfg,
            'val_acc_mean': mean_acc,
            'val_acc_std': std_acc,
            'val_accs': val_accs
        })
        
        print(f"  Mean: {mean_acc:.4f} (±{std_acc:.4f})\n")
    results.sort(key=lambda x: x['val_acc_mean'], reverse=True)

    print("\n--- Grid search result ---")
    print(f"{'Rank':<4} {'Mean accuracy':<10} {'Std':<10} {'LR':<8} {'WD':<10} {'DIMS'}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['val_acc_mean']:.4f}     ±{r['val_acc_std']:.4f}    {r['lr']:<8} {r['wd']:<10} {r['hidden_dims']}")

    best = results[0]
    print(f"\n Best combination: LR={best['lr']}, WD={best['wd']}, Dims={best['hidden_dims']}")
    print(f"   Mean accuracy: {best['val_acc_mean']:.4f} (±{best['val_acc_std']:.4f})")

def visualize_loss(loss_record):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_record, linewidth=2, color='blue', marker='o', markersize=4, markevery=5)
    plt.xlabel('Total step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_val_loss(val_loss_record):
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss_record, linewidth=2, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Curve')
    plt.grid(True)
    plt.show()

def visualize_acc(acc_record):
    plt.figure(figsize=(10, 6))
    plt.plot(acc_record, linewidth=2, color='blue', marker='o', markersize=4, markevery=5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training accuracy Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hidden_dims = (512, 256, 128)
    lr = 0.1
    wd = 0.0001

    acc_record, loss_record, val_loss_record= run(hidden_dims=hidden_dims, epochs=10, lr=lr, weight_decay=wd)
    visualize_loss(loss_record)
    visualize_val_loss(val_loss_record)
    visualize_acc(acc_record)
