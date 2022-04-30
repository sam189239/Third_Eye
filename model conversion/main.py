from Focus_mode import everydayobjectdetection,find_object_area
vidpath = 'data/everydayobject_test11.mp4'
object_areas = 'objects_areas.npy'
class_video = 'classes/classes.mp4'
width = 640  # video resolutions
height = 480 
res = (width,height)

start_focus_mode = True
register_new_classes = False

if start_focus_mode == True:
    detection = everydayobjectdetection(object_threshold = object_areas,show_video=True)
    detection.detect_object(videopath= vidpath, res = res)
elif register_new_classes == True:
    # update the object bounding box area at 2 feet to calculate different object thresholds.
    object_area = find_object_area()
    object_area.object_area(class_video,res)


