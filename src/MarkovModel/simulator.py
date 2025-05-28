# main.py

import numpy as np
import time
import signal
import random
from event_manager import EventManager

# --- Configurable Parameters ---
frequencies = [0.5, 1.0, 1.5, 2.0]
thresholds = [60, 70, 50, 80]
loop_delay = 0.1  # seconds

# --- State ---
active_times = [None, None, None, None]
last_remote_event_time = 0

# --- Graceful Exit Setup ---
running = True
def signal_handler(sig, frame):
    global running
    print("\nInterrupted! Exiting gracefully...")
    running = False
signal.signal(signal.SIGINT, signal_handler)

# --- Event Manager Instance ---
event_manager = EventManager()

# --- Remote Event Simulation Function ---
def simulate_remote_event(current_time):
    global last_remote_event_time

    # One remote event per second max
    if current_time - last_remote_event_time >= 1.0:
        if random.random() < 0.3:  # 30% chance every second
            canvas_id = random.choice(["rhythm", "script"]) # the local canvas instead is "Echoes"
            channel_id = random.randint(1, 4)
            event_manager.handle_remote_event(canvas_id, channel_id)
            last_remote_event_time = current_time

# --- Main Loop ---
print("Running threshold monitor with EventManager. Press Ctrl+C to stop.")
start_time = time.time()

while running:
    current_time = time.time() - start_time
    sine_values = [int((np.sin(2 * np.pi * f * current_time) + 1) * 50) for f in frequencies]

    for i, value in enumerate(sine_values):
        if value > thresholds[i]:
            if active_times[i] is None:
                active_times[i] = current_time
            elapsed = current_time - active_times[i]
            event_manager.handle_local_event(i + 1, elapsed, value)
        else:
            active_times[i] = None

    simulate_remote_event(current_time)

    time.sleep(loop_delay)

print("Done.")

