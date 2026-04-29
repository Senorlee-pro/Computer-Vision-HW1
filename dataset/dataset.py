import numpy as np
from torchvision import datasets

class Dataloader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = len(self.dataset)
        self.images, self.labels = self.load_data()

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        if len(self.images.shape) == 3:
                self.images = self.images.reshape(self.images.shape[0], -1)


    def load_data(self):
        images = []
        labels = []

        for i in range(self.dataset_len):
            img, label = self.dataset[i]
            img_np = np.array(img)
            img_flat = img_np.reshape(-1)
            img_flat = img_flat / 255.0
            images.append(img_flat)
            labels.append(label)
        return images, labels
    
    def __iter__(self):
        indices = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.dataset_len, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.dataset_len)
            current_idx = indices[start_idx : end_idx]
            image_batch = self.images[current_idx]
            labels_batch = self.labels[current_idx]

            yield image_batch, labels_batch

    def __len__(self):
        return self.dataset_len // self.batch_size + 1
    
class DatasetSubset:
    def __init__(self, full_dataset, indices):
        self.full_dataset = full_dataset
        self.indices = indices

    def __getitem__(self, i):
        return self.full_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)
    
