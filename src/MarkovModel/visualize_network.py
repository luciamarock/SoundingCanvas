# simulator_with_network_20s.py

import numpy as np
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
from event_manager import EventManager  # your HOMM class

# --- Configurable Parameters ---
frequencies = [0.5, 1.0, 1.5, 2.0]
thresholds = [60, 70, 50, 80]
loop_delay = 0.1  # seconds
sound_ids = list(range(8))  # Assuming 8 possible sounds
simulation_duration = 0.6  # seconds

# --- State ---
active_times = [None] * len(frequencies)
last_remote_event_time = 0
running = True

# --- Event Manager ---
event_manager = EventManager()

# --- Remote event simulation ---
def simulate_remote_event(current_time):
    global last_remote_event_time
    if current_time - last_remote_event_time >= 1.0:
        if random.random() < 0.3:  # 30% chance per second
            canvas_id = random.choice(["rhythm", "script"])
            channel_id = random.randint(1, 4)
            event_manager.handle_remote_event(canvas_id, channel_id)
            last_remote_event_time = current_time

# --- Main loop ---
print("Running HOMM simulation for 20 seconds...")
start_time = time.time()

while running:
    current_time = time.time() - start_time
    if current_time >= simulation_duration:
        running = False
        break

    # Simulate sensor values (sine wave oscillation)
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

print("Simulation finished. Building Network Graph...")

# --- Build Network Graph ---
G = nx.DiGraph()
for context, transitions in event_manager.transition_counts.items():
    for next_sound, count in transitions.items():
        G.add_edge(str(context), str(next_sound), weight=count)

# Draw the graph
plt.figure(figsize=(12,8))
ax = plt.gca()  # Get current axes

pos = nx.spring_layout(G, seed=42)
weights = [G[u][v]['weight'] for u,v in G.edges()]
edges = nx.draw(
    G, pos, ax=ax, with_labels=False, node_size=200, 
    edge_color=weights, edge_cmap=plt.cm.viridis, width=2, arrowsize=15
)

sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                           norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
sm.set_array([])
#plt.colorbar(sm, ax=ax, label='Transition count')  # Explicitly provide ax
plt.title('HOMM Transition Network')
plt.show()

