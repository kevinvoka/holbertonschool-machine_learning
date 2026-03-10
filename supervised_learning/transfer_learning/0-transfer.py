#!/usr/bin/env python3
"""Transfer Learning: Train CIFAR-10 classifier using EfficientNetB0."""

import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-processes CIFAR-10 data for the EfficientNetB0-based model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 images.
        Y: numpy.ndarray of shape (m,) containing CIFAR-10 class labels.

    Returns:
        X_p: numpy.ndarray of preprocessed images (float32, values in [0, 255]
             scaled per EfficientNet requirements).
        Y_p: numpy.ndarray of one-hot encoded labels with shape (m, 10).
    """
    X_p = K.applications.efficientnet.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    #  Hyperparameters                                                     #
    # ------------------------------------------------------------------ #
    IMG_SIZE = 224       # EfficientNetB0 optimal resolution
    BATCH_SIZE = 256
    PHASE1_EPOCHS = 20   # train only the top head (features cached)
    PHASE2_EPOCHS = 30   # fine-tune top layers of base model
    FINE_TUNE_AT = -30   # unfreeze last 30 base-model layers

    # ------------------------------------------------------------------ #
    #  Data loading & preprocessing                                        #
    # ------------------------------------------------------------------ #
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)

    # ------------------------------------------------------------------ #
    #  Base model (frozen)                                                 #
    # ------------------------------------------------------------------ #
    base_model = K.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    base_model.trainable = False

    # Build a feature-extractor that resizes 32x32 → IMG_SIZE x IMG_SIZE
    # then passes through the frozen base.  We use Lambda so the saved
    # model remains self-contained (tf.image.resize is available in the
    # Keras deserialisation namespace).
    _input = K.Input(shape=(32, 32, 3))
    _resized = K.layers.Lambda(
        lambda x: tf.image.resize(x, (IMG_SIZE, IMG_SIZE)),
        name='resize'
    )(_input)
    _feats = base_model(_resized, training=False)
    feature_extractor = K.Model(_input, _feats, name='feature_extractor')

    # ------------------------------------------------------------------ #
    #  Hint 3: compute frozen-layer outputs ONCE (cache features)         #
    # ------------------------------------------------------------------ #
    print("Caching training features through frozen base model...")
    X_train_feats = feature_extractor.predict(
        X_train_p, batch_size=BATCH_SIZE, verbose=1
    )
    print("Caching validation features...")
    X_val_feats = feature_extractor.predict(
        X_val_p, batch_size=BATCH_SIZE, verbose=1
    )
    feat_shape = X_train_feats.shape[1:]  # e.g. (1280,) with pooling='avg'

    # ------------------------------------------------------------------ #
    #  Phase 1 — train only the classification head on cached features    #
    # ------------------------------------------------------------------ #
    feat_input = K.Input(shape=feat_shape, name='cached_features')
    x = K.layers.Dense(512, activation='relu')(feat_input)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.3)(x)
    head_out = K.layers.Dense(10, activation='softmax')(x)
    top_model = K.Model(feat_input, head_out, name='top_model')

    top_model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n=== Phase 1: Training classification head ===")
    top_model.fit(
        X_train_feats, Y_train_p,
        epochs=PHASE1_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_feats, Y_val_p),
        callbacks=[
            K.callbacks.EarlyStopping(
                patience=6, restore_best_weights=True, monitor='val_accuracy'
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, verbose=1
            ),
        ],
        verbose=1
    )

    # ------------------------------------------------------------------ #
    #  Assemble final end-to-end model                                     #
    # ------------------------------------------------------------------ #
    final_input = K.Input(shape=(32, 32, 3), name='image_input')
    x = K.layers.Lambda(
        lambda img: tf.image.resize(img, (IMG_SIZE, IMG_SIZE)),
        name='resize'
    )(final_input)
    x = base_model(x, training=False)
    final_output = top_model(x)
    model = K.Model(final_input, final_output, name='cifar10_transfer')

    # ------------------------------------------------------------------ #
    #  Phase 2 — fine-tune top layers of base model                       #
    # ------------------------------------------------------------------ #
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Data augmentation applied on the fly during fine-tuning
    augment = K.Sequential([
        K.layers.RandomFlip('horizontal'),
        K.layers.RandomRotation(0.1),
        K.layers.RandomZoom(0.1),
        K.layers.RandomTranslation(0.1, 0.1),
    ], name='augmentation')

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_p, Y_train_p))
        .shuffle(50000, reshuffle_each_iteration=True)
        .batch(128)
        .map(
            lambda imgs, labels: (augment(imgs, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val_p, Y_val_p))
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("\n=== Phase 2: Fine-tuning top layers of EfficientNetB0 ===")
    model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        callbacks=[
            K.callbacks.EarlyStopping(
                patience=8, restore_best_weights=True, monitor='val_accuracy'
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3,
                min_lr=1e-8, verbose=1
            ),
            K.callbacks.ModelCheckpoint(
                'cifar10.h5', save_best_only=True,
                monitor='val_accuracy', verbose=1
            ),
        ],
        verbose=1
    )

    # ------------------------------------------------------------------ #
    #  Save & evaluate                                                     #
    # ------------------------------------------------------------------ #
    model.save('cifar10.h5')
    loss, acc = model.evaluate(X_val_p, Y_val_p, batch_size=128, verbose=1)
    print(f"\nFinal validation accuracy: {acc:.4f}  (target: ≥0.87)")
