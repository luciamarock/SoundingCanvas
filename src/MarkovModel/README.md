# Gesture-to-Sound Simulation Demo

This folder contains a demonstration of the **Event Manager** and the **gesture simulation system** used in the _Sounding Canvas_ project. It models local and remote touch events using an adaptive High-Order Markov Model to drive sound selection based on gesture history.

## Files

- `simulator.py`  
  Simulates a local Sounding Canvas called `echoes`, producing four synthetic sine wave-based "touch signals" (representing capacitive sensor readings). When these exceed configured thresholds, touch events are generated and sent to the Event Manager. It also simulates random remote touch events from two other canvases: `rhythm` and `script`.

- `event_manager.py`  
  Contains the `EventManager` class, which receives and processes both local and remote events. It builds and maintains a high-order Markov model of gesture sequences, and uses it to select sounds adaptively from a per-channel set of sound options.

## Requirements

- Python 3.7+
- Standard Python libraries only (no external dependencies)

## Usage

To run the simulation:

```bash
python3 simulator.py
```


You can interrupt the loop at any time using `Ctrl+C`.

### Adjustable Parameters

Inside `simulator.py`, you can modify:

* Sine wave frequencies and thresholds
* Time between simulation cycles (`SLEEP_TIME`)
* Probability and rate of remote events
* Duration of local gestures

### Output

The system prints:

* Local events when sensor values exceed thresholds, showing:

  * Channel ID
  * Duration since activation
  * Raw signal value
* Remote events with:

  * Canvas ID
  * Channel ID

The adaptive Markov model uses this sequence of events to estimate the probability of playing each possible sound (indexed `0` to `7`) per channel.

## Concept

This demo is part of the broader *Sounding Canvas* project, where user interactions with visual artworks are translated into sound events in a spatially coherent and temporally adaptive way. Here, we simulate the behavior of the canvas in isolation and its interaction with remote canvases via a decentralized model.

## Notes

* This demo does not play actual audio, but integrates the logic needed to select and trigger audio files.
* For integration with real hardware (capacitive sensors, sound output), see the higher-level system in the main project.

---

For questions or contributions, please refer to the main repository README or contact the project maintainer.



