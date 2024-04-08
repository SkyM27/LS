from fastapi.responses import JSONResponse
import pandas as pd
import os
import uvicorn
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()


class Prediction(BaseModel):
    input: List[float]


knn = None
clf = None


@app.on_event("startup")
def load_train_model():
    df = pd.read_csv("./iris_cleaned.csv")
    X, y = df[[column for column in df.columns if column not in ["variety", "variety_encoded", "X", "Y"]]], df["Y"]

    global knn

    knn = KNeighborsClassifier(n_neighbors=len(df["Y"].unique()))

    knn.fit(X, y)

    print("Training done!")


@app.on_event("startup")
def load_anomaly_model():
    from pyod.models.knn import KNN
    df = pd.read_csv("./iris_ok.csv")
    X, y = df[[column for column in df.columns if column not in ["variety", "variety_encoded", "X", "Y"]]], df["Y"]
    global clf
    clf = KNN()

    clf.fit(X, y)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def load_model(prediction: Prediction):
    if len(prediction.input) != 4:
        return JSONResponse(status_code=400, content="Input length must be 4!")
    predictions = knn.predict([prediction.input])
    return {"prediction": int(predictions[0])}


@app.get("/anomaly")
async def load_model(p1: float, p2: float, p3: float, p4: float):
    predictions = clf.predict([[p1, p2, p3, p4]])
    return "Nu este outlier!" if predictions[0] == 0 else "Este outlier!"


if __name__ == '__main__':
    uvicorn.run(app, host=os.getenv("HOST"), port=os.getenv("PORT"))