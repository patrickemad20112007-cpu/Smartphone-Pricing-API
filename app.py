from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. بنفتح المطعم بتاعنا (تأسيس الـ API)
app = FastAPI(title="Smartphone Pricing API")

# 2. بندخل الشيف المطبخ (بنصحي الموديل بتاعنا من الفايل اللي حفظناه)
model = joblib.load('smartphone_pricing_model.pkl')

# 3. بنعمل "المنيو" اللي بيحدد شكل الطلب عشان العميل ميدخلش بيانات غلط (Validation)
class MobileFeatures(BaseModel):
    Brand: str
    RAM_GB: int
    Storage_GB: int
    Battery_mAh: int
    Camera_MP: int
    Screen_Size_inch: float

# 4. دي الترابيزة اللي الجرسون هيستقبل عليها الطلبات (الـ Endpoint)
@app.post("/predict")
def predict_price(features: MobileFeatures):
    # بناخد الطلب من العميل ونحوله لـ DataFrame (جدول) لأن الشيف بتاعنا مبيفهمش غير جداول
    # لو ظهرلك إيرور هنا في المستقبل، ممكن تبدل model_dump بـ dict
    data_dict = features.model_dump() 
    df = pd.DataFrame([data_dict])
    
    # الشيف (الموديل) بيحسب السعر، وبنحط [0] عشان نطلع الرقم الصافي من المصفوفة
    predicted_price = model.predict(df)[0]
    
    # الجرسون بيرجعلك النتيجة النهائية (الرد بيكون بصيغة JSON)
    return {
        "Message": "تم حساب السعر بنجاح",
        "Brand": features.Brand,
        "Expected_Price_USD": round(predicted_price, 2)
    }