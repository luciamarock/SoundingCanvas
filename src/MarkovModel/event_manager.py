import random
from collections import defaultdict, deque
import time

class EventManager:
    def __init__(self, order=8, decay=0.95):
        self.order = order
        self.decay = decay
        self.history = deque(maxlen=order)
        self.transition_counts = defaultdict(lambda: defaultdict(float))
        self.last_active_times = {}  # For tracking durations of local events

    def _get_context_key(self):
        return tuple(self.history)

    def _update_transition_counts(self, context, sound_id):
        # Apply decay
        for s in self.transition_counts[context]:
            self.transition_counts[context][s] *= self.decay
        # Update count
        self.transition_counts[context][sound_id] += 1.0

    def _predict_next_sound(self, context):
        counts = self.transition_counts.get(context)
        if not counts:
            # Back-off strategy
            for k in range(self.order - 1, 0, -1):
                sub_context = context[-k:]
                sub_counts = self.transition_counts.get(tuple(sub_context))
                if sub_counts:
                    counts = sub_counts
                    break
        if not counts:
            # Fallback to uniform
            return random.randint(0, 7)

        total = sum(counts.values())
        r = random.uniform(0, total)
        cumulative = 0.0
        for sound_id, count in counts.items():
            cumulative += count
            if r <= cumulative:
                return sound_id
        return random.choice(list(counts.keys()))

    def handle_local_event(self, channel_id, active_since, value):
        event = ("local", channel_id, round(active_since, 2))
        self.history.append(event)

        # Predict and select next sound ID using adaptive Markov model
        context = self._get_context_key()
        selected_sound = self._predict_next_sound(context)

        # Update the model with the new event
        self._update_transition_counts(context, selected_sound)

        # Play the selected sound (placeholder)
        print(f"[LOCAL] Channel {channel_id} active for {active_since:.2f}s → Play sound {selected_sound}")

    def handle_remote_event(self, canvas_id, channel_id):
        event = ("remote", canvas_id, channel_id)
        self.history.append(event)
        print(f"[REMOTE] Received event from canvas '{canvas_id}' channel {channel_id}")

# This class implements a High Order Markov Model (HOMM)
# to dynamically select audio events based on past local and remote interactions.
# It learns transition probabilities in real-time and adapts via decay to recent activity.

