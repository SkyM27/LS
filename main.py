from fastapi import FastAPI
from sklearn.neighbors import KNeighborsClassifier
from pyod.models.knn import KNN
import numpy as np
import pandas as pd

app = FastAPI()

neigh = None

@app.get("/")

@app.on_event("startup")
def load_train_model():
    df = pd.read_csv("iris_ok.csv")
    global neigh
    global clf
    neigh = KNeighborsClassifier(n_neighbors=len(np.unique(df['y'])))
    neigh.fit(df[df.columns[:4]].values.tolist(), df["y"])
    clf = KNN()
    clf.fit(df[df.columns[:4]].values.tolist(), df["y"])
    print("Model finished the training")

@app.get("/predict")
def predict(p1: float, p2: float, p3: float, p4: float):
    pred = neigh.predict([[p1, p2, p3, p4]])
    return "{}".format(pred[0])

@app.get("/anomaly")
def anomaly(p1: float, p2: float, p3: float, p4: float):
    pred = clf.predict([[p1, p2, p3, p4]])
    return "{}".format(pred[0])

async def root():
    return {"Message": "Hello World"}