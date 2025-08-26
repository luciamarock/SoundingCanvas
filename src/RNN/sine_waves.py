import numpy as np
import random 

class WavesGenerator:
    def __init__(self):
        self._sampling_frequency = 1000
        self._frequency = 100
        self._amplitude = 2

    def generate_values(self,number_of_samples):
        wave_values = []
        for i in range(number_of_samples):
            current_time = i / self._sampling_frequency
            wave_values.append(np.sin(2 * np.pi * self._frequency * current_time + 1) * self._amplitude)
    
        return wave_values

    def detect_gesture_speed(self, touch_time):
        number_of_samples = int(self._sampling_frequency * touch_time)
        wave_values = self.generate_values(number_of_samples)
        speed = self.detect_speed(wave_values)
        return speed
    
    def detect_speed(self,wave_values):
        # speed in cm/sec
        #TODO implement this 
        return random.randint(0, 7)
    
    def close(self):
        print("closing simulator data stream")