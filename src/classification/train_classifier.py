import os
import optuna
import random
import tensorflow as tf
import data_utils
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

img_size = (224, 224)
batch_size = 32
seed = 42

trials = 24  
head_run_epochs = 6  
head_final_epochs = 12 
finetune_epochs = 12  
checkpoint_file = "densenet_best.h5"
final_model_file = "densenet_final.h5"
unfreeze_last_n = 30 
# --------------------------------------------
train_ds, val_ds, classes = data_utils.get_datasets(
    data_utils.src_folder, data_utils.patient_classes
)


def build_model(num_classes, dropout_rate, head_units, backbone_trainable=False):
    inp = Input(shape=img_size + (3,))
    aug = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
        ]
    )
    x = aug(inp)
    x = layers.Lambda(densenet_preprocess)(x)

    backbone = DenseNet121(
        include_top=False, weights="imagenet", input_shape=img_size + (3,)
    )
    backbone.trainable = backbone_trainable
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(head_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    return model, backbone


### head till now ###
def compile_and_train(
    model, train_ds, val_ds, lr, epochs, ckpt_path, patience=3, verbose=1
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=verbose
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=verbose
    )
    return history


def objective(trial, train_ds, val_ds, num_classes):
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    head_units = trial.suggest_categorical("head_units", [64, 128, 256])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)

    model, _ = build_model(num_classes, dropout, head_units, backbone_trainable=False)
    history = compile_and_train(
        model,
        train_ds,
        val_ds,
        lr,
        head_run_epochs,
        ckpt_path="/tmp/optuna_ckpt.h5",
        patience=2,
        verbose=0,
    )

    val_accs = history.history.get("val_accuracy", [])
    best_val = max(val_accs) if val_accs else 0.0
    return best_val


def run_optuna_search(
    train_ds,
    val_ds,
    num_classes,
    db_path="sqlite:///optuna_study.db",
    csv_path="optuna_trials.csv",
):
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=db_path,
        study_name="densenet_opt_study",
        load_if_exists=True,
    )

    def save_trial(study_obj, trial_obj):
        df = study_obj.trials_dataframe()
        df.to_csv(csv_path, index=False)

    study.optimize(
        lambda t: objective(t, train_ds, val_ds, num_classes),
        n_trials=trials,
        callbacks=[save_trial],
    )

    print("Best trial params:", study.best_trial.params)
    return study.best_trial.params


def unfreeze_backbone(backbone, last_n):
    # slast_n trainable
    total = len(backbone.layers)
    start = max(0, total - abs(last_n))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= start


def train_with_optuna():
    num_classes = len(classes)
    best = run_optuna_search(train_ds, val_ds, num_classes)
    dropout = best["dropout"]
    head_units = best["head_units"]
    best_lr = best["lr"]

    model, backbone = build_model(
        num_classes, dropout, head_units, backbone_trainable=False
    )
    # train head
    compile_and_train(
        model,
        train_ds,
        val_ds,
        best_lr,
        head_final_epochs,
        ckpt_path=checkpoint_file,
        patience=3,
        verbose=1,
    )

    # unfreeze backbone
    if os.path.exists(checkpoint_file):
        model.load_weights(checkpoint_file)

    unfreeze_backbone(backbone, unfreeze_last_n)
    compile_and_train(
        model,
        train_ds,
        val_ds,
        lr=1e-5,
        epochs=finetune_epochs,
        ckpt_path=checkpoint_file,
        patience=3,
        verbose=1,
    )

    if os.path.exists(checkpoint_file):
        final_model, _ = build_model(
            num_classes, dropout, head_units, backbone_trainable=True
        )
        final_model.load_weights(checkpoint_file)
        final_model.save(final_model_file)
        # print(final_model_file)
        return final_model_file

    model.save(final_model_file)
    return final_model_file


if __name__ == "__main__":
    out = train_with_optuna()
    # print("->", out)
