import time
import math
from openal.audio import SoundSink, SoundSource
from openal.loaders import load_wav_file
from openal.al import *
from openal.alc import *

if __name__ == "__main__":
    sink = SoundSink()
    sink.activate()
    source1 = SoundSource(position=[0, 0, 0])
    source2 = SoundSource(position=[0, 0, 4])
    source1.looping = True
    source2.looping = True
    data = load_wav_file("bounce.wav")
    source1.queue(data)
    source2.queue(data)
    # sink.play(source1)
    sink.play(source2)
    t = 0
    while True:
        # x_pos = 5*math.sin(math.radians(t))
        # source1.position = [abs(x_pos), source1.position[1], source1.position[2]]
        # source2.position = [-1 * abs(x_pos), source1.position[1], source1.position[2]]
        sink.update()
        print("playing at %r" % source1.position)
        time.sleep(0.1)
        t += 5
