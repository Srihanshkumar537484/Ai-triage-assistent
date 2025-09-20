from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware # Nayi line

# 1. FastAPI app banayein
app = FastAPI()

# Naya Code: CORS middleware jodein
origins = [
    "http://127.0.0.1:5500",  # Agar aap Live Server istemal kar rahe hain
    "http://localhost:5500",  # Agar aap Live Server istemal kar rahe hain
    "*"  # Ya fir sabhi origins ki anumati dein
]

# Naya Code: Sabhi origins ki anumati dein (sirf local testing ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Is line ko badlein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 2. Apne trained model aur vectorizer ko load karein
model = joblib.load("triage_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# 3. User ke input data ke liye ek model banayein
class TriageRequest(BaseModel):
    symptoms: str

# 4. API endpoint banayein
@app.post("/predict")
def predict_triage(request: TriageRequest):
    # a. User ke symptoms ko ek list mein daalein
    symptoms_list = [request.symptoms]
    
    # b. Vectorizer ka istemal karke text ko numbers mein badlein
    symptoms_vectorized = vectorizer.transform(symptoms_list)
    
    # c. Model se prediction karwayein
    prediction = model.predict(symptoms_vectorized)
    predicted_disease = prediction[0]
    
    # d. Prediction ko JSON format mein return karein
    return {"predicted_disease": predicted_disease}