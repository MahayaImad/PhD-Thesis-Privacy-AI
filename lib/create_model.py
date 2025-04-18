import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_LIST = {
    "convnext": tf.keras.applications.ConvNeXtSmall,
    "densenet": tf.keras.applications.DenseNet121,
    "efficientnet": tf.keras.applications.EfficientNetB0,
    "efficientnet_v2": tf.keras.applications.EfficientNetV2S,
    "inception_resnet_v2": tf.keras.applications.InceptionResNetV2,
    "inception_v3": tf.keras.applications.InceptionV3,
    "mobilenet": tf.keras.applications.MobileNet,
    "mobilenet_v2": tf.keras.applications.MobileNetV2,
    "mobilenet_v3": tf.keras.applications.MobileNetV3Small,
    "nasnet": tf.keras.applications.NASNetMobile,
    "resnet": tf.keras.applications.ResNet101,
    "resnet50": tf.keras.applications.ResNet50,
    "resnet_v2": tf.keras.applications.ResNet152V2,
    "vgg16": tf.keras.applications.VGG16,
    "vgg19": tf.keras.applications.VGG19,
    "xception": tf.keras.applications.Xception,
}

def create_model(architecture="efficientnet", 
                 input_shape=(224, 224, 3), 
                 num_classes=9, 
                 dropout_rate=0.3, 
                 dense_units=128,
                 learning_rate = 1e-4,
                 optimizer = "adam",
                 freeze_base=True):
    """
    By : MAHAYA IMAD (15-04-2025)
    We create a transfer learning model with a specified architecture and diferent hyperparameters.
    
    Args:
        architecture (str): CNN architecture name (that exists already in Keras).
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes in dataset.
        dropout_rate (float): Dropout rate before the final layer.
        dense_units (int): Number of units in the Dense layer before the output.
        freeze_base (bool): If True, base model weights are frozen (Transfer Learning on imageNet).
        optimizer (str): Type of optimizer when building model.
        learning_rate (float): learning rate that we initiate optimizer with.
    """
    architecture = architecture.lower()
    if architecture not in MODEL_LIST:
        raise ValueError(f"Unfound architecture '{architecture}'.")

    base_model_chosed = MODEL_LIST[architecture]
    base_model = base_model_chosed(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )

    # Transfer learning
    base_model.trainable = not freeze_base

    x = base_model.output
    if dense_units > 0:
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    if optimizer == "adam":
      opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
      opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
      opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
      optimizer=opt,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
    )

    return model