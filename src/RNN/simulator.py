import time 
import datetime
import random
import matplotlib.pyplot as plt
import json
import sine_waves


exhibition_date = datetime.datetime(2025, 8, 18, 16, 11, 00)
exhibition_start_epoch_time = int(exhibition_date.timestamp())
#print(exhibition_start_epoch_time)
#print(datetime.datetime.fromtimestamp(exhibition_start_epoch_time))
SECONDS_IN_HOUR = 3600
HOUR_OF_EXHIBITION = 8
MAXIMUM_INACTIVITY_SECONDS = 900
MAXIMUM_ACTIVITY_SECONDS = 650
MAXIMUM_LISTENING_OF_SOUND = 25

NUMBER_OF_CHANNELS = 4
SOUNDS_PER_CHANNEL = 8


simulator = sine_waves.WavesGenerator()

TOTAL_EXHIBITION_DURATION_SECONDS = SECONDS_IN_HOUR * HOUR_OF_EXHIBITION
elapsed_seconds = 0

data = {}
data["sessions"] = []


x = []
ch = []
snd = []

while elapsed_seconds < TOTAL_EXHIBITION_DURATION_SECONDS:
    session_dict = {}
    inactivity_period = random.randint(0, MAXIMUM_INACTIVITY_SECONDS)
    elapsed_seconds += inactivity_period
    activity_period = random.randint(0, MAXIMUM_ACTIVITY_SECONDS)
    interaction_cycle_finish_time = elapsed_seconds + activity_period
    events = []
    prev_sound_id = None
    prev_pause = None
    is_init = True
    while elapsed_seconds < interaction_cycle_finish_time:
        channel_id = random.randint(0, NUMBER_OF_CHANNELS-1) + 1
        sound_id = random.randint(0, SOUNDS_PER_CHANNEL-1) + 1
        sound_number = (channel_id - 1) * SOUNDS_PER_CHANNEL + sound_id
        event_dict = {}
        event_dict["sound_id"] = prev_sound_id
        prev_sound_id = sound_number
        event_dict["channel_id"] = channel_id
        pause = random.randint(2, MAXIMUM_LISTENING_OF_SOUND)
        event_dict["delta_t"] = prev_pause
        prev_pause = pause 
        elapsed_seconds+=pause 
        touch_time = pause / random.uniform(2.0, 10.0)
        average_speed = simulator.detect_gesture_speed(touch_time)
        event_dict["touch_time"] = touch_time
        event_dict["average_speed"] = average_speed
        if is_init:
            is_init = False
        else:
            events.append(event_dict)
        #x.append(elapsed_seconds + exhibition_start_epoch_time)
        #ch.append(channel_id)
        #snd.append(sound_number)
    if events:
        session_dict["events"] = events 
        data["sessions"].append(session_dict)
    #print(divmod(elapsed_seconds,SECONDS_IN_HOUR))

filename = "data.json"
with open(filename, 'w') as f:
    json.dump(data,f,indent=4)
#plt.plot(x,ch,'o')
#plt.plot(x,snd,'x')
#plt.show()
