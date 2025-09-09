import tensorflow as tf
from tensorflow.keras import layers
from utils.patient_split import patient_wise_split   # NEW import

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def get_datasets(source_dir, output_dir="patient_split", train_ratio=0.8):
    # Perform patient-wise split once
    train_dir, val_dir = patient_wise_split(source_dir, output_dir, train_ratio, SEED)

    # Load datasets from split folders
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names
