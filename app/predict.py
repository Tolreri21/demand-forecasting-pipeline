import numpy as np
from pydantic import BaseModel
from typing import List


class InputData(BaseModel):
    CountryIndex: float
    StockCodeIndex: float
    Month: int
    Year: int
    DayOfWeek: int
    Day: int
    Week: int

class OutputData(BaseModel):
    prediction: float


def predict_from_input(model, input_data: InputData) -> OutputData:
    if model is None:
        raise ValueError("Model is not loaded")

    X = np.array([[
        input_data.CountryIndex,
        input_data.StockCodeIndex,
        input_data.Month,
        input_data.Year,
        input_data.DayOfWeek,
        input_data.Day,
        input_data.Week
    ]])
    y_pred = model.predict(X)[0]
    return OutputData(prediction=float(y_pred))