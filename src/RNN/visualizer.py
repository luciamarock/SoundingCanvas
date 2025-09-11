import json
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
with open('data.json', 'r') as f:
    data = json.load(f)

# -----------------------
# Flatten all events with absolute time
# -----------------------
all_events = []
current_time = 0
for session in data['sessions']:
    events = session['events']
    for e in events:
        current_time += e['delta_t'] / 10.0  # scale delta_t to seconds (adjust as needed)
        all_events.append({
            'abs_time': current_time,
            'sound_id': e['sound_id'],
            'channel_id': e['channel_id'],
            'average_speed': e['average_speed']
        })
    current_time += 5  # add gap between sessions to separate visually

# -----------------------
# Scatter plot per event
# -----------------------
plt.figure(figsize=(15,5))
colors = [e['channel_id'] for e in all_events]  # color by channel
sizes = [5 + 20*e['average_speed'] for e in all_events]  # size by speed

plt.scatter([e['abs_time'] for e in all_events],
            [e['sound_id'] for e in all_events],
            c=colors, s=sizes, cmap='tab10', alpha=0.7)

plt.xlabel('Absolute Time (s)')
plt.ylabel('Sound ID')
plt.title('Per-Event Timeline of All Sessions')
plt.colorbar(label='Channel ID')
plt.show()

