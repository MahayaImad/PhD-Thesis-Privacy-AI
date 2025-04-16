import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import urllib.request
import gc

class PathMNISTLoader:
    def __init__(self, 
                 dataset_dir='datasets', 
                 filename='pathmnist', 
                 num_classes=9, 
                 batch_size=32):
        
        self.dataset_dir = dataset_dir
        self.filepath = os.path.join(dataset_dir, filename + '.npz')  # Ensure .npz extension
        self.download_url = f"https://zenodo.org/records/10519652/files/{filename}.npz?download=1"
        self.num_classes = num_classes
        self.batch_size = batch_size

        os.makedirs(dataset_dir, exist_ok=True)
        self._download_if_needed()
        self._load_data()  # Load in memory (you can modify this later for full on-disk lazy loading)

    def _download_if_needed(self):
        if not os.path.exists(self.filepath):
            print(f"Downloading dataset to {self.filepath}...")
            urllib.request.urlretrieve(self.download_url, self.filepath)
            print("Download complete.")

    def _load_data(self):
        data = np.load(self.filepath, mmap_mode='r')  # mmap_mode for memory efficiency
        print("Loaded data keys:", data.files)

        self.x_train = data['train_images']
        self.y_train = data['train_labels']
        self.x_val = data['val_images']
        self.y_val = data['val_labels']
        self.x_test = data['test_images']
        self.y_test = data['test_labels']

        print("Train:", self.x_train.shape, self.y_train.shape)
        print("Validation:", self.x_val.shape, self.y_val.shape)
        print("Test:", self.x_test.shape, self.y_test.shape)

    def _preprocess(self):
        # Normalize and one-hot encode in slices to save memory
        self.x_train = self.x_train / 255.0
        self.x_val = self.x_val / 255.0
        self.x_test = self.x_test / 255.0

        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_val = to_categorical(self.y_val, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def _batch_generator(self, data, labels):
        for start in range(0, len(data), self.batch_size):
            end = min(start + self.batch_size, len(data))
            yield data[start:end], labels[start:end]

    def _create_tf_dataset(self, x_data, y_data):
        input_shape = x_data.shape[1:]
        dataset = tf.data.Dataset.from_generator(
            lambda: self._batch_generator(x_data, y_data),
            output_signature=(
                tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32)
            )
        )
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def get_datasets(self):
        self._preprocess()

        train_ds = self._create_tf_dataset(self.x_train, self.y_train)
        val_ds = self._create_tf_dataset(self.x_val, self.y_val)
        test_ds = self._create_tf_dataset(self.x_test, self.y_test)

        # Free memory
        del self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test
        gc.collect()

        return train_ds, val_ds, test_ds