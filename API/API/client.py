import requests

filename = "recipt.jpg"
files = {"file":('recipt.jpg', open("recipt.jpg", 'rb'))}
print(files)
response = requests.post(
	'http://13.58.19.126:8000/document_mode/images',
	files = files,
	)

print(response.json())