import cv2 
import time
import math
from openal.audio import SoundSink, SoundSource
from openal.loaders import load_wav_file

from json.decoder import JSONDecodeError
import json
import time

json_file_path = "obs_state.json"
with open(json_file_path, 'r') as j:
            contents = json.loads(j.read())
            obs_hist = contents['state']

# Alert sound options:
# pop
# bounce
# droplet 13, 18
# beep

sink = SoundSink()
sink.activate()

source = [SoundSource(position=[-5, 0, 0]),SoundSource(position=[5, 0, 0]), SoundSource(position=[0, 0, 5])] ## Left, Right, Mid

source[0].looping = True
source[1].looping = True
source[2].looping = True

data = load_wav_file("sounds/beep-07.wav")

source[0].queue(data)
source[1].queue(data)
source[2].queue(data)

while True:
    try:
        with open(json_file_path, 'r') as j:
            contents = json.loads(j.read())

            obs_current = contents['state']
            time.sleep(0.3)
            for a in [0,1,2]:
                if obs_current[a]:
                    sink.play(source[a])
                    sink.update()
                else:
                    sink.pause(source[a])
                    sink.update()

    except JSONDecodeError:
        continue