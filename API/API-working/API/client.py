import requests

print("@@@@@ Document Mode @@@@@")
filename = "recipt.jpg"
files = {"file":('recipt.jpg', open("recipt.jpg", 'rb'))}
print(files)
response = requests.post(
	'http://127.0.0.1:8000/document_mode/images',
	files = files,
	)

print(response.json())


print("#########")
print("@@@@@ Simple  Mode @@@@@")

filename = "recipt.jpg"
files = {"file":('recipt.jpg', open("recipt.jpg", 'rb'))}
print(files)


response = requests.post(
        'http://127.0.0.1:8000/abstract_mode/images/',
        files = files,
        )

print(response.json())

print("#########")
print("@@@@@ Currency  @@@@@")

filename = "recipt-10.jpg"
files = {"file":('recipt-10.jpg', open("recipt-10.jpg", 'rb'))}
print(files)


response = requests.post(
        'http://127.0.0.1:8000/currency/images/',
        files = files,
        )

print(response.json())

print("#########")
print("@@@@@  FaceReco  @@@@@")

filename = "recipt-ananth_detect.jpg"
files = {"file":('recipt-ananth_detect.jpg', open("recipt-ananth_detect.jpg", 'rb'))}
print(files)


response = requests.post(
        'http://127.0.0.1:8000/facereco/images/',
        files = files,
        )

print(response.json())
