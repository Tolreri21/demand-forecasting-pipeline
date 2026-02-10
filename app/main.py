import uvicorn
import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager

from predict import predict_from_input, InputData, OutputData

PATH = "app/models/RandomForest.pkl"

model = None


def load_model(path):
    global model
    model = joblib.load(path)


def unload_model():
    global model
    model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(PATH)
    yield
    unload_model()


app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=OutputData)
def predict(input_data: InputData):
    return predict_from_input(model, input_data)

PORT = 8000

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT )
