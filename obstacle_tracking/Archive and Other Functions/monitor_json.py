from json.decoder import JSONDecodeError
import json
from bottle import run, post, request, response
from playsound import playsound
import numpy as np
import time

json_file_path = "obs_state.json"
with open(json_file_path, 'r') as j:
            contents = json.loads(j.read())
            obs_hist = contents['state']

safe_report = False

while True:

    try:
        with open(json_file_path, 'r') as j:
            contents = json.loads(j.read())

            obs_current = contents['state']
            
            ## Left, Mid, Right, Safe Warnings ## Checking for state change every frame - affected by delays ##
            if np.sum(obs_current) == 0 and np.sum(obs_hist)!=0: # No obstacle, present in prev frame
                playsound('sounds/safe.mp3')
                print("Safe")
            elif obs_current[2] == 1 and obs_hist[2] == 0: # Obstacle in the mid, not in prev frame
                playsound('sounds/warning.mp3')
                print("Mid Warning")
            elif (obs_current[0] and obs_current[1]) and ((obs_hist[1] != obs_current[1]) or (obs_hist[0] != obs_current[0])): # Obstacle in both left and right but absent in either or both in prev frame
                playsound('sounds/warning.mp3')
                print("LR Warning")
            elif obs_current[0] == 1 and obs_hist[0] == 0: # Obstacle in Left alone, absent in prev frame
                playsound('sounds/left.mp3')
                print("Left Warning")
            elif obs_current[1] == 1 and obs_hist[1] == 0: # Obstacle in right alone, absent in prev frame
                playsound('sounds/right.mp3')
                print("Right Warning")

            obs_hist = obs_current

            ## Left, Mid, Right, Safe Warnings ## Checking every 2 seconds - reports current state except repeated safe ##
            # time.sleep(2)
            # if np.sum(obs_current) == 0 and not safe_report: # No obstacle, present in prev frame
            #     playsound('sounds/safe.mp3')
            #     print("Safe")
            #     safe_report = True
            # elif obs_current[2] == 1: # Obstacle in the mid
            #     playsound('sounds/warning.mp3')
            #     print("Mid Warning")
            #     safe_report = False
            # elif (obs_current[0] and obs_current[1]): # Obstacle in both left and right 
            #     playsound('sounds/warning.mp3')
            #     print("LR Warning")
            #     safe_report = False
            # elif obs_current[0] == 1: # Obstacle in Left alone
            #     playsound('sounds/left.mp3')
            #     print("Left Warning")
            #     safe_report = False
            # elif obs_current[1] == 1: # Obstacle in right alone
            #     playsound('sounds/right.mp3')
            #     print("Right Warning")
            #     safe_report = False

            # obs_hist = obs_current

    except JSONDecodeError:
        # print("error")
        continue