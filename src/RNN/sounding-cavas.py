#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:54:29 2025
This is the equivalent of event_manager.py in MarkovModel directory

@author: luciamarock
"""
import time
import random 
import numpy as np
import tensorflow as tf

NUM_CHANNELS = 4
SOUNDS_PER_CHANNEL = 8
MAXIMUM_COUNTS = 16000
MAXIMUM_LISTENING_OF_SOUND = 25

class MachineAnswer:
    """
    The 'machine' of the Sounding Canvas.
    Uses the trained model to choose the next sound
    based on the user's interaction history and a goal channel.
    """

    def __init__(self, model_path, num_channels=NUM_CHANNELS, sounds_per_channel=SOUNDS_PER_CHANNEL):
        # Load trained keras model
        self.model = tf.keras.models.load_model(model_path)
        self.max_seq_len = 150 # same as the model max_seq_len
        # Configuration
        self.num_channels = num_channels
        self.sounds_per_channel = sounds_per_channel
        self.current_channel_ID = None
        self.last_touch_time = time.time()
        self.candidate_delta_t = 0.0001

        # Keep history of (sound_id, delta_t)
        self.history = {}
        self.history["sound_ids"] = []
        self.history["delta_ts"] = []
        self.history["touch_times"] = []
        self.history["touch_speeds"] = []
        self.channel_count = {}
        for i in range(NUM_CHANNELS):
            self.channel_count[str(i + 1)] = 0

    def reset_history(self):
        """Clear the stored interaction history."""
        self.candidate_delta_t = 0.0001
        self.history = {}
        self.history["sound_ids"] = []
        self.history["delta_ts"] = []
        self.history["touch_times"] = []
        self.history["touch_speeds"] = []
        for i in range(NUM_CHANNELS):
            self.channel_count[str(i + 1)] = 0
    
    def _analyze_gesture(self):
        #TODO implement this 
        touch_time = random.uniform(2.0, 10.0)
        average_speed =random.randint(0, 7)
        return touch_time, average_speed

    def register_user_touch(self, channel_id):
        """
        Register a new user event in the history.
        :param sound_id: int
        :param delta_t: float
        """
        self.current_channel_ID = channel_id
        self.candidate_delta_t = time.time() - self.last_touch_time
        if self.candidate_delta_t >= MAXIMUM_LISTENING_OF_SOUND:
            self.reset_history()
        self.last_touch_time = time.time()
        self.channel_count[str(channel_id)] += 1
        touch_time, touch_speed = self._analyze_gesture()
        sound_to_play = self.decide_next_sound(touch_time, touch_speed)
        return sound_to_play
    
    def _create_goal(self):
        min_count = MAXIMUM_COUNTS
        for key, value in self.channel_count.items():
            if value < min_count:
                min_count = value
                channel_ID = int(key)
            if value >= MAXIMUM_COUNTS:
                self.reset_history()
        return channel_ID

    def _predict_next_channel(self, sound_ids, delta_ts, touch_times, avg_speeds):
        """
        Runs model prediction given a full history sequence.
        Returns a probability distribution over channels.
        """
        #print(sound_ids)
        #print(delta_ts)
        # Pad or truncate the sequence to the model's expected length
        pad_len = self.max_seq_len - len(sound_ids)
        if pad_len > 0:
            sound_ids = [0] * pad_len + sound_ids
            delta_ts = [0.0] * pad_len + delta_ts
            touch_times = [0.0] * pad_len + touch_times
            avg_speeds = [0.0] * pad_len + avg_speeds
        else:
            sound_ids = sound_ids[-self.max_seq_len:]
            delta_ts = delta_ts[-self.max_seq_len:]
            touch_times = touch_times[-self.max_seq_len:]
            avg_speeds = avg_speeds[-self.max_seq_len:]
    
        # Features: shape (seq_len, 3)
        features = np.stack([delta_ts, touch_times, avg_speeds], axis=-1)
            
        # Create a batch of size 1 for prediction
        X_sounds = np.array([sound_ids])
        X_features = np.array([features])     # shape (1, seq_len, 3)
        
        # Model expects a tuple of inputs: (sounds, times)
        probs = self.model.predict((X_sounds, X_features), verbose=0)[0] # shape (num_channels,)
        
        return probs

    def decide_next_sound(self,touch_time, touch_speed):
        """
        Decide which sound to play next to maximize
        the probability that the user touches goal_channel_id.
        """
        
        goal_channel_id = self._create_goal()
        #print("Machine wants the user to touch channel {} next".format(goal_channel_id))
        # Evaluate all candidate sounds across all channels
        best_sound = None
        best_prob = -1.0

        sound_range_start = (self.current_channel_ID - 1) * self.sounds_per_channel + 1
        sound_range_end = sound_range_start + self.sounds_per_channel
        #print("-------------- loop begins -----------------")
        self.history["touch_times"].append(touch_time)
        self.history["touch_speeds"].append(touch_speed)
        possible_solutions = []
        for candidate_sound in range(sound_range_start, sound_range_end):
            # For now, assume delta_t = 0 for machine-generated sound
            h_sound_ids = self.history["sound_ids"].copy()
            h_sound_ids.append(candidate_sound) #hypotetical
            h_delta_ts = self.history["delta_ts"].copy()
            h_delta_ts.append(self.candidate_delta_t) #hypotetical
            h_touch_times = self.history["touch_times"].copy()
            h_touch_speeds = self.history["touch_speeds"].copy()
            probs = self._predict_next_channel(h_sound_ids,h_delta_ts,h_touch_times,h_touch_speeds)
            #print(f"channel probabilities for candidate sound {candidate_sound} are {probs}")
            max_index = np.argmax(probs)
            if max_index + 1 == goal_channel_id:
                possible_solutions.append(candidate_sound)
            prob_goal = probs[goal_channel_id - 1]
            #print(f"probability of channel goal {goal_channel_id} is {prob_goal}")
            if prob_goal > best_prob:
                best_prob = prob_goal
                best_sound = candidate_sound
        #print("-------------- loop ends -----------------")
        #print("best sound is {} with probability to make the goal happen of {}".format(best_sound,best_prob))
        possible_solutions.append(best_sound)
        final_selection = random.choice(possible_solutions)
        self.history["sound_ids"].append(final_selection)
        self.history["delta_ts"].append(self.candidate_delta_t)

        return final_selection

if __name__ == "__main__":
    machine = MachineAnswer("next_touch_given_sounds.keras")
    start_time = time.time()
    current_time = time.time()
    while current_time - start_time < 120:
        channel_ID = random.randint(1,NUM_CHANNELS)
        sound_ID = machine.register_user_touch(channel_ID)
        print(f"machine answered with {sound_ID} to user touch at {channel_ID}")
        sleep_time = random.randint(1000,10000)
        time.sleep(sleep_time/1000.)
        current_time = time.time()