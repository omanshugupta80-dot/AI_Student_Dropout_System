import joblib
import pandas as pd

# Load model
model = joblib.load("model/trained_model.pkl")

def predict_student(student_data):
    df = pd.DataFrame([student_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability