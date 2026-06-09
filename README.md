# ⚡ Electricity Consumption Predictor

## 📌 Project Overview
This project predicts the next hour’s electricity consumption using machine learning models trained on historical household electricity usage data. The application is deployed using Streamlit for real-time prediction.
<img width="940" height="442" alt="image" src="https://github.com/user-attachments/assets/bab5730d-7a9f-4107-b171-d44cb68f1c98" />


## 🚀 Live Application
👉 https://electricity-usage-predictor.streamlit.app/

## 🧠 Features
- Time-aware forecasting
- Lag-based prediction (previous usage)
- Rolling mean feature
- Real-time prediction using Streamlit UI
- Clean and interactive interface

## 🛠 Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit

## ⚙️ Machine Learning Workflow
1. Data cleaning and handling missing values
2. Resampling minute-level data to hourly
3. Feature engineering:
   - Hour, Day
   - Lag features (lag_1, lag_2)
   - Rolling mean
4. Feature scaling using StandardScaler
5. Feature selection using Forward Selection
6. Model training:
   - Ridge Regression
   - Lasso Regression
   - PCR
   - PLS
7. Validation using TimeSeriesSplit
8. Evaluation using Mean Squared Error (MSE)

## 📊 Model Evaluation
The models were compared based on MSE, and the best-performing model was selected for deployment.

## ▶️ Run Locally
```bash
streamlit run app.py
