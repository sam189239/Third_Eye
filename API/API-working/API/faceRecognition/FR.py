from keras_facenet import FaceNet 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity


directory = r"data\ananth"
test_image_path = r"data\ananth.jpg"
alpha = 2
embedder = FaceNet()
database = {}
font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade= cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

def image_path(directory):
	image_path = []
	dirs = os.listdir(directory)
	for d in dirs:
		image_path.append(os.path.join(directory,d))
	print("No of images found:",len(image_path))
	return image_path 


images_path = image_path(directory)
print(images_path)

def Faces(paths,single = False):
	''' it will detect the face in a image 
	    and will return a list of cropped images containing only face.
	    
	    input: list of image path on working directory

	'''

	crop_img = []

	
	if not single:
		for path in paths:
			img = cv2.imread(path)
		#		img = cv2.resize(img , (96,96) , interpolation = cv2.INTER_NEAREST)
		#   	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			face = faceCascade.detectMultiScale(img,1.1,4)
			
			if len(face) != 0:
				for (x,y,w,h) in face:
					Cropped = img[y : y + h , x: x+w ]
					Cropped = cv2.cvtColor(Cropped , cv2.COLOR_BGR2RGB)
					crop_img.append(Cropped)
	else:
		img = cv2.imread(paths)
		
		#		img = cv2.resize(img , (96,96) , interpolation = cv2.INTER_NEAREST)
		#   	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		face = faceCascade.detectMultiScale(img,1.1,4)
			
		if len(face) != 0:
			
			for (x,y,w,h) in face:
				Cropped = img[y : y + h , x: x+w ]
				Cropped = cv2.cvtColor(Cropped , cv2.COLOR_BGR2RGB)
				crop_img.append(Cropped)

	    
	return crop_img

faces = Faces(images_path)


embeddings = embedder.embeddings(faces)

database["ananth"] = embeddings


def Face_recog(image_path , alpha = 2):
    '''
    image_path : list of individual input image path.
    alpha : it is a hyperparameter
    '''
    
    #detecting face
    print("detecting...")
    image = Faces(image_path, single = True)
    
    print("loading facenet...")
    img_embedding = embedder.embeddings(image)
    
    #calculate dist wrt to database images
    min_dist = 100
    for (name,db_emb) in database.items():
        for emb in db_emb:
        	dist = np.linalg.norm(img_embedding - emb)
        	if dist < min_dist:
        		min_dist = dist
        		identity = name
    if min_dist > alpha:
    	print("Not found in Database, Kindly contact the admin!!")
    else:
    	print("distance:",min_dist)
    	return identity


def imshow(identity,paths):
	img = cv2.imread(paths)
	img = cv2.resize(img , (512,512) , interpolation = cv2.INTER_NEAREST)
	face = faceCascade.detectMultiScale(img,1.1,4)
	
	if len(face) != 0:
			
		for (x,y,w,h) in face:
			box_img = cv2.rectangle(img,(x,y+h),(x+w ,y),(0,255,0),1)
			box_img = cv2.putText(box_img,identity,(x+w+1,y+h+1),font,0.5,(255,0,255),1)
			

		cv2.imshow("Face recognition", box_img)
		cv2.waitKey(0)


identity = Face_recog(test_image_path)
imshow(identity,test_image_path)


