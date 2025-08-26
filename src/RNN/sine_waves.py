import numpy as np
import time

class WavesGenerator:
    def __init__(self, number_of_channels, multiplier, amplitude):
        f0 = 0.5  # Hz
        self._amplitude = amplitude/2
        self._frequencies = [f0 * (multiplier ** i) for i in range(number_of_channels)]
        self._start_time = time.time()

    def generate_next_values(self):
        current_time = time.time() - self._start_time
        sine_values = [
            int((np.sin(2 * np.pi * f * current_time) + 1) * self._amplitude)
            for f in self._frequencies
        ]
        return sine_values

    def readline(self):
        values = self.generate_next_values()
        line = "\t".join(map(str, values)) + "\n"
        return line.encode("utf-8")  # simulate byte string like from serial port

    def close(self):
        print("closing simulator data stream")