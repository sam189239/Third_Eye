import pyttsx3 as tts
import json
from bottle import run, post, request, response

engine = tts.init()

@post('/process')
def my_process():
  req_obj = json.loads(request.body.read())
  # do something with req_obj
  # ...
  print(req_obj)
#   if req_obj["warn"]:
#     engine.say("Warning")
#     engine.runAndWait()
  return 'All done'

run(host='localhost', port=8080, debug=True)
# 'All done'


