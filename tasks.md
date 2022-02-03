 documentation and comments - X
 
 threshold for slopes check necessity - X
 missing obstacle handling
 removing prev values if change is too much
 global threshold on size - X
 tracking readme - X
 avg the warnings over n frames - X
 unique thresholds - X
 wider fov - inform - X

 inc warning avg - X
 obj at extremities - X
 obj at the center - X
 soft threshold for angle of objects in ROI - X
 inner roi with bigger threshold - no angle condition - X

 person coming closer - X
 del angle threshold based on position of obstacle - X
 anomaly removal

 VERY BIG OBJECTS IN THE MIDDLE OF ROI

 alert for left right and mid, l+r - X
 sudden variations due to missing bounding boxes - X
 multithreading - X
 remove extremity warning - X

 removing color change for outside roi - X
 check averaging - X
 remove outside roi threshold condition - X
 safe sound - X


 remove warn_db - X
 combine obstacle detection and tracking folders, add readme - X

 Current progress:

TTS based alert system
Modified warning system by detecting obstacle state
Limited alert counts
Set up alert system for live demo

9/1/22
 worked on thresholds
 warning through file handling

10/1/22
 tested new client video
 reduced warning averaging
 good results on warning algo
 set up alerting through 3d sound

upto 18/1/22
 reduced false alarms
 tested options for alert sound
 cleaned up code

upto 24/1/22
 added documentation
 added comments for all functions
 moved the secondary functions to a different script
 created merge request with master branch

 merging warn and alert
 footpath
 webcam test

upto 3/2/2022
 Set up single script with warn and alert but has a minor delay
 Explored deployment options like Jetson Nano, Google Coral Edge TPU board
 Merged Obstacle tracking module with other modules
 Added subprocess implementation allowing calling another script in parallel, hence, no delay
 Finalized single function needed for module to be run
 Added requirements.txt for all modules
 updated gitignores, readmes, running documentation
