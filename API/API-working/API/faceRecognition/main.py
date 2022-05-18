from FR2 import face_recognition

train_dir = [r"data\ananth",r"data\prashant"]
test_image_path = r"data\ananth.jpg"
threshold = 1
database = "database.npy"
faceCascade= "haarcascades/haarcascade_frontalface_default.xml"
users = ["ananth", "prashant"]

face_recog = face_recognition(threshold= threshold, haarcascades = faceCascade,
	database_exist = True, database_path = database)

face_recog.face_detection(source = "webcam")