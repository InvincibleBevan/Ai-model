import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

# Load env vars if .env exists
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/KurmaAI/AQUA-1B")

app = FastAPI(title="AQUA-1B Water Quality Inference")

class Reading(BaseModel):
    deviceId: str
    pH: float
    turbidity: float
    temperature: float
    conductivity: float

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# You may need to use another pipeline/task depending on model type!
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

def format_input(data: Reading):
    # Change input format as required by AQUA-1B prompt schema
    return (
        f"Device: {data.deviceId}\n"
        f"pH: {data.pH}\n"
        f"Turbidity: {data.turbidity}\n"
        f"Temperature: {data.temperature}\n"
        f"Conductivity: {data.conductivity}"
    )

@app.post("/predict")
def predict(reading: Reading):
    text = format_input(reading)
    result = pipe(text)[0]
    # Return and map as needed: risk_score, label, anomaly
    risk_score = float(result.get("score", 0.0))
    label = result.get("label", "UNKNOWN")
    # Customize anomaly threshold for your use case!
    anomaly = bool(risk_score > 0.7)
    return {
        "risk_score": risk_score,
        "label": label,
        "anomaly": anomaly
    }

@app.get("/")
def root():
    return {"status": "ok"}
