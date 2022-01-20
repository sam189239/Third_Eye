import cv2 
import numpy as np

def footpath(image):

	upperLeftPoint = [220,270]
	lowerLeftPoint = [143,417]
	upperRightPoint = [440,270]
	lowerRightPoint = [630,417]

	pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
	lowerRightPoint]], dtype=np.int32)

	masked_image = region_of_interest(image, pts)

	HSV = cv2.cvtColor(masked_image,cv2.COLOR_BGR2HSV)
	lower_gray = np.array([0, 0, 19])
	upper_gray = np.array([179, 61, 127])
	mask = cv2.inRange(HSV, lower_gray, upper_gray)
	result = cv2.bitwise_and(image, image, mask=mask)

	return result

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with 
    #depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image