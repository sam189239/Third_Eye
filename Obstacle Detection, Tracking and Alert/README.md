## Obstacle Detection, Tracking and Alerting 

Files:
- warn_json.py -> Outputs the obstacle state for every frame to obs_state.json (in use)
- alert_3dsound.py -> Constantly reads the JSON file and starts and stops audio source as alerts for the user at left, right and mid -> Almost no delay (in use)
- functions.py -> Contains the functions used in the main program, imported by warn_json.py

- merged.py -> Combines warn_json.py and alert_3dsound.py at the cost of a minor delay

- warn_sound.py -> Directly alerts through playsound module, has some delays (not used)
- monitor_json.py -> Constantly reads the JSON file and plays alert sound when obs_state changes -> still involves delays as the state monitor doesn't capture changes when the audio is being played (not used)


To get inference:
- Run alert_3dsound.py
- In another terminal, run warn_json.py

Overall algorithm of program:

- Frames are collected from source and sent to models for object detection.
- Deepsort and YOLO models are initialized to obtain bounding boxes and object id.
- Output is passed to detect_obs to process obstacles in the frame
- detect_obs performs the main algorithm to detect whether an object is a potential obstacle to the user.
- Confidence threshold is applied.
- Initially, objects within the external ROI are taken into consideration along with application of size threshold.
- Obstacle is entered into the obstacle database.
- Area and angle calculations are made to check if they are increasing or decreasing.
- Increasing area and decreasing angle is an indicator for 
- A warning deque is maintained for each obstacle where TRUE or FALSE is appended for each frame.
- Alert and Color change are invoked when more than half of the warning deque is TRUE.
- This is updated in the obs and obs_current arrays. obs -> count of obsatacles in left, right and mid. obs_current -> boolean array indicating obstacle presence in left, right and mid
- Previous obstacle state is stored in obs_hist and any variation from this is updated byu sending current state to a JSON file obs_state.json which is used by alert_3dsound.py to give audio alerts.
- An additional overall warning circle is also added to the image output
- Final output is saved
