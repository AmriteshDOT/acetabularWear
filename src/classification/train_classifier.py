import os
import optuna
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_utils import get_datasets, get_augmentation, IMG_SIZE
from model_utils import build_densenet_head, build_final_model
from plot_utils import plot_history

DATA_DIR = "combined"
EPOCHS_HEAD = 10
EPOCHS_FINE = 10
CHECKPOINT_PATH = "best_densenet_head.h5"

# ----- Load Data -----
train_ds, val_ds, class_names = get_datasets(DATA_DIR)
NUM_CLASSES = len(class_names)
data_augmentation = get_augmentation()


# ----- Optuna Objective -----
def objective(trial):
    model = build_densenet_head(trial, IMG_SIZE, NUM_CLASSES, data_augmentation)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
        ],
        verbose=0,
    )
    return max(history.history.get("val_accuracy", [0.0]))


# ----- Hyperparameter Search -----
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_trial.params
print("Best hyperparameters:", best_params)

# ----- Final Model (Head Training) -----
model = build_final_model(best_params, IMG_SIZE, NUM_CLASSES, data_augmentation)
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stopping, checkpoint],
)

# ----- Fine-tune last layers -----
for layer in model.layers[-20:]:
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=[early_stopping, checkpoint],
)

# ----- Evaluate -----
model.load_weights(CHECKPOINT_PATH)
val_loss, val_acc = model.evaluate(val_ds)
print(f"Final validation accuracy: {val_acc:.4f}")

# ----- Plot -----
plot_history(history_head, "Head-only training")
plot_history(history_finetune, "Fine-tuning")

# ----- Save -----
model.save("densenet_finetuned.h5")
