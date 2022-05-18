from FR2 import face_recognition

database = "database.npy"
threshold = 1
faceCascade= "haarcascades/haarcascade_frontalface_default.xml"

face_recog = face_recognition(threshold= threshold, haarcascades = faceCascade,
	database_exist = True, database_path = database)

face_recog.train(name = "obama",img_count = 5)

