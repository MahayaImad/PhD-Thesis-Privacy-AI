import os
import tensorflow as tf

def get_callbacks(
    model_name = 'best_model',
    checkpoint_dir='models',
    tensorboard_log_dir='logs',
    monitor='val_loss',
    patience=5,
    reduce_lr_factor=0.5,
    min_lr=1e-6
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f'{model_name}.keras'),
        monitor=monitor,
        save_best_only=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=3,
        min_lr=min_lr
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1
    )
    return [early_stopping, checkpoint, tensorboard]

    # return [early_stopping, checkpoint, reduce_lr, tensorboard]
