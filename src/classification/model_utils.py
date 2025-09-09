import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

def build_densenet_head(trial, img_size, num_classes, data_augmentation):
    backbone = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,)
    )
    backbone.trainable = False

    inputs = Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = layers.Lambda(densenet_preprocess_input)(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
    head_units = trial.suggest_categorical("head_units", [64, 128, 256])
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)

    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(head_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_final_model(best_params, img_size, num_classes, data_augmentation):
    backbone = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,)
    )
    backbone.trainable = False

    inputs = Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = layers.Lambda(densenet_preprocess_input)(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(best_params["dropout_rate"])(x)
    x = layers.Dense(best_params["head_units"], activation="relu")(x)
    x = layers.Dropout(best_params["dropout_rate"])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
