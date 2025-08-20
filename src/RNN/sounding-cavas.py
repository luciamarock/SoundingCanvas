#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 09:54:29 2025

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
        self.channel_count = {}
        for i in range(NUM_CHANNELS):
            self.channel_count[str(i + 1)] = 0

    def reset_history(self):
        """Clear the stored interaction history."""
        self.candidate_delta_t = 0.0001
        self.history = {}
        self.history["sound_ids"] = []
        self.history["delta_ts"] = []
        for i in range(NUM_CHANNELS):
            self.channel_count[str(i + 1)] = 0

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
        #print("user touched channel {}, computing sound ...".format(channel_id))
        sound_to_play = self.decide_next_sound()
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

    def _predict_next_channel(self,sound_ids,delta_ts):
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
        else:
            sound_ids = sound_ids[-self.max_seq_len:]
            delta_ts = delta_ts[-self.max_seq_len:]
            
        # Create a batch of size 1 for prediction
        X_sounds = np.array([sound_ids])
        X_times = np.array([delta_ts])
        
        # Model expects a tuple of inputs: (sounds, times)
        probs = self.model.predict((X_sounds, X_times), verbose=0)[0] # shape (num_channels,)
        return probs

    def decide_next_sound(self):
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
        for candidate_sound in range(sound_range_start, sound_range_end):
            # For now, assume delta_t = 0 for machine-generated sound
            h_sound_ids = self.history["sound_ids"].copy()
            h_sound_ids.append(candidate_sound) #hypotetical
            h_delta_ts = self.history["delta_ts"].copy()
            h_delta_ts.append(self.candidate_delta_t) #hypotetical
            probs = self._predict_next_channel(h_sound_ids,h_delta_ts)
            #print(f"channel probabilities for candidate sound {candidate_sound} are {probs}")
            prob_goal = probs[goal_channel_id - 1]
            #print(f"probability of channel goal {goal_channel_id} is {prob_goal}")
            if prob_goal > best_prob:
                best_prob = prob_goal
                best_sound = candidate_sound
        #print("-------------- loop ends -----------------")
        #print("best sound is {} with probability to make the goal happen of {}".format(best_sound,best_prob))

        self.history["sound_ids"].append(best_sound)
        self.history["delta_ts"].append(self.candidate_delta_t)

        return best_sound

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