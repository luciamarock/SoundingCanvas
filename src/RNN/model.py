#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:54:29 2025

@author: luciamarock
"""
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

NUM_CHANNELS = 4
SOUNDS_PER_CHANNEL = 8

class SessionDataset(tf.keras.utils.Sequence):
    """
    This is an implementation of a custom data generator.
    It inherits from tf.keras.utils.Sequence, which is a standard
    and highly recommended way to feed data to a Keras model.

    It turns data.json into sequences of (sound_id, Δt) with labels = next channel_id.
    Each session is a sequence until abandonment.
    """
    def __init__(self, path, max_seq_len=50, batch_size=32):
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        """
        Build (input_sequence, label) pairs from sessions.
        Each input sequence = list of (sound_id, delta_t) up to time i,
        Label = channel_id at time i.
        """
        samples = []
        for session in self.data["sessions"]:
            events = session["events"] # list of {sound_id, delta_t, channel_id}
            for i, ev in enumerate(events):
                # inputs = all past (sound_id, delta_t) up to this event
                hist = [{
                    "sound_id": e["sound_id"],
                    "delta_t": e["delta_t"],
                    "touch_time": e.get("touch_time", 0.0),
                    "average_speed": e.get("average_speed", 0.0)
                } for e in events[:i+1]]
                # label = channel_id of this event
                label = ev["channel_id"] - 1
                samples.append((hist, label))
        return samples

    def __len__(self):
        """
        This method returns the number of batches in the dataset.
        Keras uses this to know how many steps to run per epoch.
        """
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        """
        This method is the core of the data generator. It's called for
        each batch index (idx) and must return a tuple of (inputs, labels).
        The inputs are a tuple because the model has two inputs.
        
        The padding logic is crucial for handling variable-length sequences,
        ensuring all sequences have the same length (max_seq_len) for the LSTM layer.
        """
        batch = self.samples[idx*self.batch_size : (idx+1)*self.batch_size]
        X_sounds, X_features, y = [], [], []
        for hist, label in batch:
            sound_ids = [e["sound_id"] for e in hist]
            features = [[e["delta_t"], e["touch_time"], e["average_speed"]] for e in hist]
            # pad or truncate sequences to max_seq_len
            pad_len = self.max_seq_len - len(sound_ids)
            if pad_len > 0:
                # Pad with zeros at the beginning
                sound_ids = [0]*pad_len + sound_ids
                features = [[0.0, 0.0, 0.0]]*pad_len + features
            else:
                # Truncate from the beginning
                sound_ids = sound_ids[-self.max_seq_len:]
                features = features[-self.max_seq_len:]
            X_sounds.append(sound_ids)
            X_features.append(features)
            y.append(label)
        
        # The return value must be a tuple of (inputs, labels)
        return (
            {
                "sound_id": np.array(X_sounds, dtype=np.int32),
                "features": np.array(X_features, dtype=np.float32)
            },
            np.array(y, dtype=np.int32)
        )


def build_model(vocab_size, embedding_dim=32, rnn_units=64, max_seq_len=50, num_channels=4, dropout_rate=0.2):
    """
    This is the Recurrent Neural Network (RNN) architecture.
    It handles two different types of input data: categorical
    (sound_id) and numerical (delta_t).

    Inputs:
        sound_input: The sequence of sound IDs. It's 'int32' because it's a list of IDs.
        time_input: The sequence of time deltas. It's 'float32' for numerical data.
    """
    # Inputs
    sound_input = layers.Input(shape=(max_seq_len,), dtype="int32", name="sound_id")
    features_input = layers.Input(shape=(max_seq_len, 3), dtype="float32", name="features")

    # Embedding for sounds:  It converts each sound_id
    # into a dense vector representation that the model can learn from.
    sound_emb = layers.Embedding(vocab_size, embedding_dim, mask_zero=False)(sound_input)

    # Concatenate embeddings + time: This combines the two feature streams
    # into a single vector for each step in the sequence.
    x = layers.Concatenate(axis=-1)([sound_emb, features_input])

    # LSTM encoder: The LSTM layer processes the sequence data, learning
    x = layers.LSTM(rnn_units)(x) 
    #x = layers.LSTM(rnn_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(x)
    #x = layers.LSTM(rnn_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)(x)

    # Output layer (predict next channel): The final Dense layer produces
    # the predictions. 
    out = layers.Dense(num_channels, activation="softmax")(x)

    # Create the model using the Functional API, which allows for multiple inputs.
    model = tf.keras.Model(inputs=[sound_input, features_input], outputs=out)
    # The compilation settings (optimizer, loss, metrics) 
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    try:
        dataset = SessionDataset("data.json", max_seq_len=150, batch_size=32)
    except FileNotFoundError:
        print("Error: 'data.json' file not found. Please create the file with the correct data format.")
        sys.exit(1)

    # Vocab_size is set based on the maximum sound_id + 1.
    # A max sound_id of 32 means a vocab_size of 33 (since IDs are 0-indexed).
    vocab_size = NUM_CHANNELS * SOUNDS_PER_CHANNEL + 1

    model = build_model(vocab_size=vocab_size, num_channels=NUM_CHANNELS, max_seq_len=150)
    model.summary()

    # Keras will automatically use __len__ to determine the number of steps.
    model.fit(dataset, epochs=11)
    model.save("next_touch_given_sounds.keras")

