from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Early Loan Closure Prediction API")

# Load the saved Random Forest model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input data structure - same 6 features you trained
class LoanData(BaseModel):
    Loan_Amount: float
    Interest_Rate: float
    Tenure_Months: int
    Customer_Age: int
    Monthly_Income: float
    Credit_Score: int

@app.get("/")
def home():
    return {"message": "API is Running - Early Loan Closure Prediction"}

@app.post("/predict")
def predict(data: LoanData):
    # IMPORTANT: Order must be same as training
    # Loan_Amount, Interest_Rate, Tenure_Months, Customer_Age, Monthly_Income, Credit_Score
    input_values = np.array([[
        data.Loan_Amount,
        data.Interest_Rate,
        data.Tenure_Months,
        data.Customer_Age,
        data.Monthly_Income,
        data.Credit_Score
    ]])

    prediction = model.predict(input_values)[0]
    probability = model.predict_proba(input_values)[0].max()

    if prediction == 1:
        result = "Customer WILL close loan early"
    else:
        result = "Customer will NOT close loan early"

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": f"{probability*100:.2f}%"
    }