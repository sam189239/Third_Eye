# Obstacle detection and warning using YOLOv5s

How to run:
- virtualenv te
- source te/bin/activate or te\Scripts\activate
- pip install -r req.txt
- store input video in data folder and change file name in script
- python "script name".py

Scripts: 
- roi -> detects objects within a region of interest
- obstacle_roi -> detects all obstacles and differentiates between those within and outside the ROI
- orw -> adds on to obstacle_roi by giving warning in left, mid and right regions of the roi if two or more obstacles of size above a threshold are present

Tune threshold as per requirement and resolution.
