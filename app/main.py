from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import asyncio

model = None
PATH = ""


def load_model():
    global model
    model = joblib.load(PATH)

def unload_model():
    global model
    model = None

@asynccontextmanager
async def lifespan():
    load_model()
    yield
    unload_model()

app = FastAPI(lifespan = lifespan)
