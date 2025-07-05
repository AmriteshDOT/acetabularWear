import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import DenseNet121  
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "combined"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Detected classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ]
)


def build_densenet_head(trial):

    backbone = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
    )
    backbone.trainable = False  # freeze for head-only training

    inputs = Input(shape=IMG_SIZE + (3,))
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
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def objective(trial):
    model = build_densenet_head(trial)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)],
        verbose=0,
    )
    val_accs = history.history.get("val_accuracy", [])
    # if len(val_accs) == 0:
    #     return 0.0
    return max(val_accs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_trial.params
print("Best hyperparameters found by Optuna:", best_params)

backbone = DenseNet121(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
backbone.trainable = False
#full final  model
inputs = Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = layers.Lambda(densenet_preprocess_input)(x)
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(best_params["dropout_rate"])(x)
x = layers.Dense(best_params["head_units"], activation="relu")(x)
x = layers.Dropout(best_params["dropout_rate"])(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)
checkpoint_path = "best_densenet_head.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping, checkpoint],
)


for layer in backbone.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping, checkpoint],
)

model.load_weights(checkpoint_path)
val_loss, val_acc = model.evaluate(val_ds)
print(f"Final validation accuracy: {val_acc:.4f}")

#Plotting curves

def plot_history(hist, title):
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

plot_history(history_head, "Head-only training")
plot_history(history_finetune, "Fine-tuning last DenseNet blocks")

# save
model.save("densenet_finetuned.h5")
