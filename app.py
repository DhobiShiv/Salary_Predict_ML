from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# CORS Fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("api_salary.pkl", "rb"))

class InputData(BaseModel):
    YearsExperience: float

@app.get("/")
def home():
    return {"message": "All Ok"}

@app.post("/predict")
def predict(data: InputData):
    exp = data.YearsExperience

    result = model.predict([[exp]])

    return {
        "YearsExperience": exp,
        "PredictedSalary": float(result[0][0])
    }