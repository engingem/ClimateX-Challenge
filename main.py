from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "ClimateX Challenge API is running!"}
