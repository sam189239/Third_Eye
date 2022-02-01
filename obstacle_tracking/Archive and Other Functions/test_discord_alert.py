import requests

url = "https://discord.com/api/webhooks/922495832421441626/LUuPSBWspEM_Sg-v1SZGZlsfSS_8U7bFnaZbcbuNI6_uKjyh512O9CB-e0hoO7qrqWWA"
label = "Person"
position = "left"
alert = f"/tts {label} {position}"

data = {'content':alert}

response = requests.post(url, data=data)
