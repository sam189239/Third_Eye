from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from pavement_tracker import footpath
# from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import time


cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
path = '../data/client_vid_1.mp4'

images = []
cap = cv2.VideoCapture(path)

fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second camera: {0}".format(fps))

# Number of frames to capture
num_frames = 1;

print("Capturing {0} frames".format(num_frames))

success = True 


counter = 0
while success:
	
	success,image = cap.read()
	if not success:
		break

	# Start time
	start = time.time()

	image = cv2.resize(image,(640,420), interpolation = cv2.INTER_AREA)
	predictions , segmentinfo = predictor(image)["panoptic_seg"]
	end = time.time()
	viz = Visualizer(image[:,:,::-1] , metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN
		[0]))
	output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"),segmentinfo)

	

	# Time elapsed
	seconds = end - start
	#print ("Time taken : {0} seconds".format(seconds))

	# Calculate frames per second
	fps  = num_frames / seconds
	print(fps)

	image  = np.array(output.get_image()[:,:,::-1])
	result = footpath(image)
	result = cv2.putText(result, "FPS: " + str(round(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255))

	# cv2.imshow("Result",result)
	images.append(result)
	# cv2_imshow(result)
	counter += 1
	print("frames processed:",counter)
cap.release()
cv2.destroyAllWindows()

#writing output video
size = images[0].shape
out = cv2.VideoWriter('footpath.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 24, (size[1],size[0]))
for i in range(len(images)):
    out.write(images[i])
out.release()