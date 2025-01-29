import tensorflow as tf

from tensorflow.keras import layers, Sequential, regularizers


def get_model(input_shape, mirrored_strategy=False):
    if mirrored_strategy:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        # Create and compile the model inside the strategy scope
        model = Sequential([
            layers.Reshape((60, 500, 1), input_shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Reshape((-1, 128)),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(128, dropout=0.3)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model