import numpy as np
import matplotlib.pyplot as plt

def visualize_fc1_from_npz(npz_path, num_show=64):
    data = np.load(npz_path)
    W = data["fc1_weight"]   # shape: (784, hidden_dim)

    W = W.T  
    W = W[:num_show]

    imgs = W.reshape(-1, 28, 28)
    cols = 8
    rows = num_show // cols

    plt.figure(figsize=(cols, rows))

    for i in range(num_show):
        plt.subplot(rows, cols, i + 1)

        img = imgs[i]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.suptitle("fc1 Weights Visualization")
    plt.tight_layout()
    plt.show()

visualize_fc1_from_npz("./best_model.npz", num_show=64)