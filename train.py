# -*- coding: utf-8 -*-
# MobileNetV2 + Transfer Learning Training Script

import tensorflow as tf
import os, json, argparse, random, numpy as np

# Đặt seed để kết quả ổn định
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

def main(args):
    data_dir = args.data_dir
    img_size = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical"
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Cải thiện tốc độ huấn luyện
    AUTOTUNE = tf.data.AUTOTUNE
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomHue(0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ])

    # --- CHANGED: apply augmentation cho train_ds ---
    def _preprocess_train(image, label):
        # Cast to float first so augmentation operations that expect float work correctly.
        image = tf.cast(image, tf.float32)
        image = data_augmentation(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    def _preprocess_val(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    train_ds = train_ds.map(_preprocess_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_preprocess_val, num_parallel_calls=AUTOTUNE)
    # ------------------------------------------------

    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Khởi tạo MobileNetV2
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet"
    )
    base.trainable = False

    # Xây mô hình
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # EarlyStopping callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    print("Training classifier (frozen base)...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Fine-tune: mở 60 lớp cuối
    base.trainable = True
    for layer in base.layers[:-105]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(7e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Fine-tuning last 105 layers...")
    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

    # Lưu model và nhãn
    model.save("seafood_model.keras")
    print("Saved seafood_model.keras")

    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)
    print("Saved labels.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
