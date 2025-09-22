# train_densenet_minimal_clean.py
import os
import optuna
import random
import tensorflow as tf
import data
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_model(num_classes, dropout_rate, head_units, backbone_trainable=False):
    inp = Input(shape=(224, 224, 3))
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
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
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


def run_optuna_search(
    train_ds,
    val_ds,
    num_classes,
    trials,
    seed,
    head_run_epochs,
    tmp_ckpt,
    db_path,
    csv_path,
):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=db_path,
        study_name="densenet_opt_study",
        load_if_exists=True,
    )

    def objective(trial):
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        head_units = trial.suggest_categorical("head_units", [64, 128, 256])
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)

        model, _ = build_model(
            num_classes, dropout, head_units, backbone_trainable=False
        )
        history = compile_and_train(
            model,
            train_ds,
            val_ds,
            lr,
            head_run_epochs,
            ckpt_path=tmp_ckpt,
            patience=2,
            verbose=0,
        )
        val_accs = history.history.get("val_accuracy", [])
        best_val = max(val_accs) if val_accs else 0.0
        return best_val

    def save_trial(study_obj, trial_obj):
        df = study_obj.trials_dataframe()
        df.to_csv(csv_path, index=False)

    study.optimize(lambda t: objective(t), n_trials=trials, callbacks=[save_trial])
    print("Best trial params:", study.best_trial.params)
    return study.best_trial.params


def unfreeze_backbone(backbone, last_n):
    total = len(backbone.layers)
    start = max(0, total - abs(last_n))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= start


def train_with_optuna(train_ds, val_ds, classes, cfg):

    num_classes = len(classes)
    best = run_optuna_search(
        train_ds,
        val_ds,
        num_classes,
        trials=cfg["trials"],
        seed=cfg["seed"],
        head_run_epochs=cfg["head_run_epochs"],
        tmp_ckpt=cfg["tmp_ckpt"],
        db_path=cfg["db_path"],
        csv_path=cfg["csv_path"],
    )
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
        cfg["head_final_epochs"],
        ckpt_path=cfg["checkpoint_file"],
        patience=3,
        verbose=1,
    )

    # unfreeze backbone
    if os.path.exists(cfg["checkpoint_file"]):
        model.load_weights(cfg["checkpoint_file"])

    unfreeze_backbone(backbone, cfg["unfreeze_last_n"])
    compile_and_train(
        model,
        train_ds,
        val_ds,
        lr=1e-5,
        epochs=cfg["finetune_epochs"],
        ckpt_path=cfg["checkpoint_file"],
        patience=3,
        verbose=1,
    )

    if os.path.exists(cfg["checkpoint_file"]):
        final_model, _ = build_model(
            num_classes, dropout, head_units, backbone_trainable=True
        )
        final_model.load_weights(cfg["checkpoint_file"])
        final_model.save(cfg["final_model_file"])
        return cfg["final_model_file"]

    model.save(cfg["final_model_file"])
    return cfg["final_model_file"]


def main():
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
    tmp_ckpt = "/tmp/optuna_ckpt.h5"
    db_path = "sqlite:///optuna_study.db"
    csv_path = "optuna_trials.csv"
    
    # cfg
    cfg = {
        "img_size": img_size,
        "batch_size": batch_size,
        "trials": trials,
        "head_run_epochs": head_run_epochs,
        "head_final_epochs": head_final_epochs,
        "finetune_epochs": finetune_epochs,
        "checkpoint_file": checkpoint_file,
        "final_model_file": final_model_file,
        "unfreeze_last_n": unfreeze_last_n,
        "tmp_ckpt": tmp_ckpt,
        "db_path": db_path,
        "csv_path": csv_path,
        "seed": seed,
    }
    train_ds, val_ds, test_ds, classes = data.get_datasets(
        data.src_folder, data.patient_classes
    )

    out = train_with_optuna(train_ds, val_ds, classes, cfg)
    # print("->", out)


if __name__ == "__main__":
    main()
