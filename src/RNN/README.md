# RNN based event-manager for Sounding Canvas

This project implements a Recurrent Neural Network (RNN) system for interactive sound selection, designed for the Sounding Canvas project. The system predicts and selects the next sound to play based on a user's interaction history, aiming to guide the user toward under-explored sound channels.

## Overview

- **model.py**: Contains the data pipeline and RNN model definition for training. It processes user session data, builds input sequences, and trains a model to predict the next channel a user will interact with.
- **sounding-cavas.py**: Implements the interactive "machine" that uses the trained model to select the next sound in real time, based on the user's recent actions.

## Data Format

The model expects a `data.json` file with the following structure:
```json
{
  "sessions": [
    {
      "events": [
        {
          "sound_id": int,
          "delta_t": float,
          "channel_id": int,
          "touch_time": float,         // optional
          "average_speed": float       // optional
        },
        ...
      ]
    },
    ...
  ]
}
```
- `sound_id`: ID of the sound played.
- `delta_t`: Time since the previous event.
- `channel_id`: Channel interacted with.
- `touch_time`, `average_speed`: Optional gesture features.

## Training the Model

1. Ensure you have `data.json` in the same directory.
2. Run:
   ```bash
   python3 model.py
   ```
   - This will train the RNN on your session data and save the model as `next_touch_given_sounds.keras`.

### Model Details

- **Inputs**: Sequences of sound IDs and gesture features (delta_t, touch_time, average_speed).
- **Architecture**: Embedding layer for sound IDs, concatenated with gesture features, processed by an LSTM, and outputting a probability distribution over channels.
- **Output**: Predicts the next channel the user is likely to interact with.

## Using the Trained Model for Sound Selection

`sounding-cavas.py` loads the trained model and simulates real-time interaction:

- The `MachineAnswer` class maintains a history of user actions and uses the model to select the next sound to play, aiming to encourage exploration of all channels.
- The script can be run directly to simulate 2 minutes of interaction:
  ```bash
  python3 sounding-cavas.py
  ```
  - The script will print the machine's sound choices in response to simulated user touches.

### Key Features

- **Adaptive Sound Selection**: The system tries to maximize the probability that the user will interact with underused channels.
- **History Management**: Resets or pads history as needed to match the model's expected input length.
- **Gesture Analysis**: (Stubbed) Placeholder for analyzing user gestures, which can be extended for richer interaction.

## Requirements

- Python 3.x
- TensorFlow
- NumPy

Install dependencies with:
```bash
pip install tensorflow numpy
```

## File Structure

- `model.py`: Data pipeline and model training.
- `sounding-cavas.py`: Real-time sound selection logic.
- `data.json`: User session data.
- `next_touch_given_sounds.keras`: Saved trained model.
