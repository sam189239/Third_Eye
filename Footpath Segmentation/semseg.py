# https://github.com/ayoolaolafenwa/PixelLib/blob/master/Tutorials/image_ade20k.md

from pixellib.semantic import semantic_segmentation
# from pixellib import semantic_segmentation
import cv2

input_dir = r"..\data\client_vid_1.mp4"

capture = cv2.VideoCapture(input_dir)

region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]    
    
cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

segment_video = semantic_segmentation()
segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segment_video.process_camera_ade20k(capture, overlay=True, frames_per_second= 15, output_video_name="output_video.mp4", show_frames= True,
frame_name= "frame")