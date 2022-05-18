import onnx
onnx_model = onnx.load("conv/yolov5s.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

width = 640  # video resolutions
height = 480 
res = (width,height)
img = cv2.imread('focus_mode/test.jpeg')
x = cv2.resize(img, res, interpolation = cv2.INTER_AREA)
# x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession("conv/yolov5s.onnx")
# outputs = ort_sess.run(None, {'input': x})
target=np.array(Image.open('focus_mode/test.jpeg'),dtype=np.int64)    
out = ort_sess.run(["out"], {"in": target.astype(np.int64)})  
# print("Container {} is empty {}, and it is classified as empty {}.".format(i, true_empty, out[0].flatten()))
print(out)
# Print Result 
# predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
# print(f'Predicted: "{predicted}", Actual: "{actual}"')


# from Focus_mode import everydayobjectdetection,find_object_area
# vidpath = 'data/everydayobject_test11.mp4'
# object_areas = 'objects_areas.npy'
# class_video = 'classes/classes.mp4'


# start_focus_mode = True
# register_new_classes = False

# if start_focus_mode == True:
#     detection = everydayobjectdetection(object_threshold = object_areas,show_video=True)
#     detection.detect_object(videopath= vidpath, res = res)
# elif register_new_classes == True:
#     # update the object bounding box area at 2 feet to calculate different object thresholds.
#     object_area = find_object_area()
#     object_area.object_area(class_video,res)
