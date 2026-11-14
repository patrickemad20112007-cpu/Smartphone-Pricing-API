# Dynamic Smartphone Pricing API 📱

An end-to-end Machine Learning project that predicts smartphone prices based on hardware specifications using a Random Forest model and a FastAPI backend.

## 🚀 Project Overview
This project demonstrates a complete Data Science workflow, from data preprocessing to model deployment. It utilizes a `scikit-learn` Pipeline to handle missing values and encode categorical features dynamically, culminating in a robust Random Forest Regressor. The trained model is saved via `joblib` and served through a high-performance RESTful API built with `FastAPI`.

## 🛠️ Technologies Used
- **Machine Learning:** Python, Scikit-Learn (Pipelines, Random Forest), Pandas, NumPy, Joblib
- **Web Framework:** FastAPI, Uvicorn, Pydantic
- **Version Control:** Git & GitHub

## ⚙️ How It Works
1. **Data Preprocessing Pipeline:** Automatically handles numerical scaling and applies One-Hot Encoding to categorical variables (like Brand) without manual intervention.
2. **Model Training:** A Random Forest Regressor trained on historical smartphone specifications.
3. **API Endpoint:** The `.pkl` model is integrated into a FastAPI application, accepting JSON payloads and returning real-time price predictions.

## 📌 API Usage
**Endpoint:** `POST /predict`

**Example Request Payload:**
```json
{
  "Brand": "Apple",
  "RAM_GB": 8,
  "Storage_GB": 256,
  "Battery_mAh": 4323,
  "Camera_MP": 48,
  "Screen_Size_inch": 6.7
}
