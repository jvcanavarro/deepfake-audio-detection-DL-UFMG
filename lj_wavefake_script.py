import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import IPython
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Reshape,
    MaxPooling2D,
    Dropout,
    Conv2D,
    MaxPool2D,
    Flatten,
)
from tensorflow.keras.utils import to_categorical
from utils.datasets import get_datasets_path

ENABLE_MIXED_PRECISION = False
MULTI_GPU = False
EXTRACT = True
DATASET_IN_CACHE = False




if ENABLE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

if MULTI_GPU:
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice()
    )
else:
    strategy = tf.distribute.get_strategy()


paths = []
labels = []

# Define the root directory
real_voice_path, fake_voice_path = get_datasets_path(DATASET_IN_CACHE)

real_root_dir = real_voice_path + "/LJSpeech-1.1/wavs"
fake_root_dir = fake_voice_path + "/generated_audio/ljspeech_melgan"

# Iterate through the subdirectories
for filename in os.listdir(real_root_dir):
    file_path = os.path.join(real_root_dir, filename)
    paths.append(file_path)
    # Add label based on the subdirectory name
    labels.append("real")

for filename in os.listdir(fake_root_dir):
    file_path = os.path.join(fake_root_dir, filename)
    paths.append(file_path)
    # Add label based on the subdirectory name
    labels.append("fake")

print("Dataset is loaded")


EXTRACT = True


def extract_features(fake_root_dir, real_root_dir, max_length=500):
    all_features = []
    labels = []

    for file in os.listdir(fake_root_dir):
        file_path = os.path.join(fake_root_dir, file)
        try:
            # Load audio file
            audio, _ = librosa.load(file_path, sr=16000)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
            mfccs = pad_or_trim(mfccs, max_length)

            # Extract Mel-Spectrogram features
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, sr=16000, n_mels=40
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram_db = pad_or_trim(mel_spectrogram_db, max_length)

            # Extract CQT features
            cqt = librosa.cqt(y=audio, sr=16000, n_bins=40)
            cqt_magnitude = pad_or_trim(np.abs(cqt), max_length)

            # Extract CQCC features (Simulated from CQT)
            cqt_db = librosa.amplitude_to_db(cqt)
            cqcc = librosa.feature.mfcc(S=cqt_db, n_mfcc=40)
            cqcc = pad_or_trim(cqcc, max_length)

            # Stack features
            combined_features = np.vstack(
                [mfccs, mel_spectrogram_db, cqt_magnitude, cqcc]
            )

            # Append features and labels
            all_features.append(combined_features)
            labels.append(1)  # 1 for fake

        except Exception as e:
            print(f"Error encountered while parsing file: {file_path}")
            continue

    for file in os.listdir(real_root_dir):
        file_path = os.path.join(real_root_dir, file)
        try:
            # Load audio file
            audio, _ = librosa.load(file_path, sr=16000)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
            mfccs = pad_or_trim(mfccs, max_length)

            # Extract Mel-Spectrogram features
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, sr=16000, n_mels=40
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram_db = pad_or_trim(mel_spectrogram_db, max_length)

            # Extract CQT features
            cqt = librosa.cqt(y=audio, sr=16000, n_bins=40)
            cqt_magnitude = pad_or_trim(np.abs(cqt), max_length)

            # Extract CQCC features (Simulated from CQT)
            cqt_db = librosa.amplitude_to_db(cqt)
            cqcc = librosa.feature.mfcc(S=cqt_db, n_mfcc=40)
            cqcc = pad_or_trim(cqcc, max_length)

            # Stack features
            combined_features = np.vstack(
                [mfccs, mel_spectrogram_db, cqt_magnitude, cqcc]
            )


            # Append features and labels
            all_features.append(combined_features)
            labels.append(0)  # 0 for real

        except Exception as e:
            print(f"Error encountered while parsing file: {file_path}")
            continue

    # Convert lists to NumPy arrays
    return np.array(all_features), np.array(labels)


def pad_or_trim(features, max_length):
    if features.shape[1] < max_length:
        return np.pad(
            features, ((0, 0), (0, max_length - features.shape[1])), mode="constant"
        )
    else:
        return features[:, :max_length]


if EXTRACT:
    x, y = extract_features(fake_root_dir, real_root_dir)
    print("Features shape:", x.shape)
    print("Labels shape:", y.shape)

np.save("x2.npy", x)
np.save("y2.npy", y)

# x = np.load('x.npy')
# y = np.load('y.npy')

split = 0.2
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=split)


with strategy.scope():
    # Create and compile the model inside the strategy scope
    model = Sequential(
        [
            layers.Reshape((160, 500, 1), input_shape=xtrain.shape[1:]),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Reshape((-1, 128)),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(128, dropout=0.3)),
            layers.BatchNormalization(),
            layers.Dense(
                128, activation="relu", kernel_regularizer=regularizers.l2(0.001)
            ),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


model.summary()


# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data and transform it
x_train_scaled = scaler.fit_transform(xtrain.reshape(-1, xtrain.shape[-1]))

# Transform the test data using the same scaler
x_test_scaled = scaler.transform(xtest.reshape(-1, xtest.shape[-1]))

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    mode="max",
    verbose=1,
)

batch_size = 32 * strategy.num_replicas_in_sync

# Train the model with early stopping
history = model.fit(
    xtrain,
    ytrain,  # Training data
    epochs=100,  # Max number of epochs
    batch_size=batch_size,  # Batch size
    validation_data=(xtest, ytest),  # Validation data
    callbacks=[early_stopping],  # Add early stopping callback
)


plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()


loss, accuracy = model.evaluate(xtest, ytest)
print(loss, accuracy)


cpu_model = tf.keras.models.clone_model(model)
cpu_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Copy weights from the TPU model to the CPU model
cpu_model.set_weights(model.get_weights())

# Save only the weights of the CPU model
cpu_model.save_weights("model.h5")
