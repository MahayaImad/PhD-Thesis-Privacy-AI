import os
import numpy as np
import urllib.request
"""
By: MAHAYA IMAD (15-04-2025)

This module imports and prepares the PathMNIST dataset.

Args:
    dataset_dir (str): Folder where dataset is saved.
    filename (str): Name of the dataset (pathmnist | pathmnist_64 | pathmnist_128 | pathmnist_224).
    num_classes (int): Number of classes (for one-hot classification).
    batch_size (int): Number of images in each batch of data.
"""

class PathMNISTLoader:
    def __init__(self, 
                 dataset_dir='datasets', 
                 filename='pathmnist', 
                 num_classes=9, 
                 batch_size=64,):
        
        self.dataset_dir = dataset_dir
        self.filepath = os.path.join(dataset_dir, filename + '.npz')
        self.download_url = f"https://zenodo.org/records/10519652/files/{filename}.npz?download=1"
        self.num_classes = num_classes
        self.batch_size = batch_size

        os.makedirs(dataset_dir, exist_ok=True)
        self._download_if_needed()
        
    def _download_if_needed(self):
        if not os.path.exists(self.filepath):
            print(f"Downloading dataset to {self.filepath}...")
            urllib.request.urlretrieve(self.download_url, self.filepath)
            print("Download complete.")


    def get_datasets(self):

        def normaliser_par_lots(array, batch_size=10000):
            n_samples = array.shape[0]
            result = []
            for i in range(0, n_samples, batch_size):
                batch = array[i:i+batch_size]
                result.append(batch.astype(np.float32) / 255.0)
            return np.concatenate(result, axis=0)

        with np.load(self.filepath) as data:
            # Entraînement
            x_train = normaliser_par_lots(data['train_images'])
            y_train = data['train_labels']
            
            # Validation
            x_val = normaliser_par_lots(data['val_images'])
            y_val = data['val_labels']
            
            # Test
            x_test = normaliser_par_lots(data['test_images'])
            y_test = data['test_labels']

        print("Train:", x_train.shape, y_train.shape)
        print("Validation:", x_val.shape, y_val.shape)
        print("Test:", x_test.shape, y_test.shape)

        return x_train , y_train, x_val, y_val, x_test, y_test
