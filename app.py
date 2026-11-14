from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title="Smartphone Pricing API")


model = joblib.load('smartphone_pricing_model.pkl')


class MobileFeatures(BaseModel):
    Brand: str
    RAM_GB: int
    Storage_GB: int
    Battery_mAh: int
    Camera_MP: int
    Screen_Size_inch: float


@app.post("/predict")
def predict_price(features: MobileFeatures):

    data_dict = features.model_dump() 
    df = pd.DataFrame([data_dict])
    
    
    predicted_price = model.predict(df)[0]
    
   
    return {
        "Message": "تم حساب السعر بنجاح",
        "Brand": features.Brand,
        "Expected_Price_USD": round(predicted_price, 2)
    }
