from keras_facenet import FaceNet 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
#import pyttsx3


class face_recognition:

	def __init__(self, threshold,haarcascades,database_path,database_exist = False,):
		 
		
		self.threshold = threshold
		self.embedder = FaceNet()
#		self.engine = pyttsx3.init()

		
		if database_exist:

			self.database = np.load(database_path , allow_pickle = True)
			self.database = self.database.tolist()
			print("database loaded!!")
		else:
			print("No database found, creating new database...")
			self.database = {}


		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.faceCascade = cv2.CascadeClassifier(haarcascades)

	def face_detection(self, source , path = None ):# None for webcam
	
		if source == "image":
			img = cv2.imread(path)
			img = cv2.resize(img , (600,800) , interpolation = cv2.INTER_NEAREST)
			
			face = self.faceCascade.detectMultiScale(img,1.1,4)
				
			if len(face) != 0:
				
				for (x,y,w,h) in face:
					Cropped = img[y : y + h , x: x+w ]
					Cropped = cv2.cvtColor(Cropped , cv2.COLOR_BGR2RGB)
					identity = self.face_recognition(Cropped)
					#print(identity)
				return identity
			# cv2.imwrite("ananth_detected.jpg",box_



		elif source == "video":
			cap = cv2.VideoCapture(path)
			success = True 
			while success:
				success , frame = cap.read()
				if success == False:
					break
				frame = cv2.resize(frame , (600,800) , interpolation = cv2.INTER_NEAREST)
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				face = self.faceCascade.detectMultiScale(frame_gray,1.3,5)
		
				if len(face) != 0:
					
					for (x,y,w,h) in face:
						Cropped = frame[y : y + h , x: x+w ]
						Cropped = cv2.cvtColor(Cropped , cv2.COLOR_BGR2RGB)
						identity = self.face_recognition(Cropped)
						print(identity)
			cap.release()
			#cv2.destroyAllWindows()
				
				


		elif source == "webcam":
			cap = cv2.VideoCapture(0)

			while True:
				success , frame = cap.read()
				if success == False:
					#self.text2speech("feed ended")
					print("feed ended")
					break

				frame = cv2.resize(frame , (600,800) , interpolation = cv2.INTER_NEAREST)
				face = self.faceCascade.detectMultiScale(frame,1.3,5)
				
				if len(face) != 0:
					
					for (x,y,w,h) in face:
						Cropped = frame[y : y + h , x: x+w ]
						Cropped = cv2.cvtColor(Cropped , cv2.COLOR_BGR2RGB)
						identity = self.face_recognition(Cropped)
						box_img = cv2.rectangle(frame,(x,y+h),(x+w ,y),(0,255,0),1)
						box_img = cv2.putText(box_img,identity,(x+w+1,y+h+1),self.font,0.5,(255,0,255),1)
			

					cv2.imshow("Face recognition", box_img)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
			cap.release()
			cv2.destroyAllWindows()
				 

		else:
			print("Not a valid source!") 


	def face_recognition(self,image):

		img_embedding = self.embedder.embeddings([image])
		min_dist = 1
		identity = "unknown"
		for (name,db_emb) in self.database.items():
			for emb in db_emb:
				dist = np.linalg.norm(img_embedding - emb)
				if dist < self.threshold:
					min_dist = dist
					identity = name
		print("minimum distance",min_dist)
		#self.text2speech(identity)			
		return identity

	def train(self,name,img_count):

		if name in self.database.keys():
			self.text2speech("Person already exists in the database. Do you wish to continue")
			user_input = input("enter Y/n")
			if user_input == "n" or user_input == 'N':
				return print('status 0')


		image_path = self.capture_images(identity = name, image_count = img_count)
		Faces = []
		image_paths = os.listdir(image_path)
		for path in image_paths:
			actual_path = os.path.join(image_path,path)
			img = cv2.imread(actual_path)
			img = cv2.resize(img , (600,800) , interpolation = cv2.INTER_NEAREST)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			face = self.faceCascade.detectMultiScale(img_gray,1.3,5)
					
			if len(face) != 0:
				
				for (x,y,w,h) in face:
					Cropped = img[y : y + h , x: x+w ]
					Cropped = cv2.cvtColor(Cropped , cv2.COLOR_BGR2RGB)
				Faces.append(Cropped)

		if len(Faces) == 0:
			self.text2speech("No Faces found!")
			return print("status 0")

		emb = self.embedder.embeddings(Faces)
		self.database[name] = emb
		np.save("database.npy",self.database)
		print("database created for: ", name)

		
	def image_path(self,path):
		image_path = []
		dirs = os.listdir(path)
		for d in dirs:
			image_path.append(os.path.join(path,d))
		print("No of images found:",len(image_path))
		return image_path 


	def text2speech(self,text):
		self.engine.say(text)
		self.engine.runAndWait()

	def blur_detection(self, image):
		blur_val = cv2.Laplacian(image, cv2.CV_64F).var()
		print(blur_val)
		if blur_val < 100:
			return True
		return False

	def capture_images(self,identity,image_count):

		cap = cv2.VideoCapture(0)
		image_storage = f"data/{identity}"
		image_counter = 0
		try:
			os.makedirs(image_storage)
		except:
			pass

		while True:

			success, frame = cap.read()
			print(success)
			if not success:
				self.text2speech("feed ended")
				break

			

			if image_counter < image_count:
				if self.blur_detection(frame):
					continue
				else:
					path = os.path.join(image_storage,identity+str(image_counter)+".jpg")
					print(path)

					cv2.imwrite(path,frame)
					image_counter = image_counter + 1
					print(image_counter)
				cv2.imshow("test",frame)
				cv2.waitKey(1)

			else:
				print("capturing complete")
				break

		return image_storage







