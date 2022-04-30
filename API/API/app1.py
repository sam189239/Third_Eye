from fastapi import FastAPI
from routers.routing import router

app = FastAPI()
app.include_router(router)


@app.get("/home/")
def welcome():
    return {"greetings": "welcome User"}


    







